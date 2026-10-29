"""通过 YAML 选择增强、去噪或原始双分支训练，复用原 SIMI 损失。"""

from collections.abc import Mapping
from copy import deepcopy
import csv
import hashlib
import math
from pathlib import Path
import shutil
import time
from typing import Optional, Union

import torch
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader

from datasets import LowLightDataset, validate_bit_mode
from datasets.sampler import ResumableBatchSampler
from losses import UnsupervisedLoss
from models import SIMI, validate_enhancement_backbone, validate_enhancement_guidance
from .validator import EpochValidator, MultiDatasetValidator
from utils.config import normalize_path, normalize_training_paths
from utils.reproducibility import capture_rng_state, restore_rng_state, seed_everything, seed_worker


class Trainer:
    """管理三种训练模式，默认 joint 保留原 train_new.py 的更新逻辑。

    enhancement：仅计算增强损失并更新增强网络，不执行去噪网络。
    denoising：冻结增强网络，在无梯度前向后仅计算去噪损失。
    joint：一次完整前向，先更新增强，再以该次前向的增强图作为
        detached 目标更新去噪；去噪损失不会回传到增强网络。

    所有训练超参数均要求显式配置。两个优化器始终保留完整参数组，
    只有当前模式启用的分支执行更新与调度，保持旧检查点结构兼容。
    """

    MODES = ("enhancement", "denoising", "joint")
    REQUIRED_FIELDS = {
        "data": ("root", "image_size", "num_workers", "shuffle", "pin_memory", "drop_last", "legacy_order"),
        "runtime": ("device", "seed", "deterministic"),
        "amp": ("enabled", "dtype", "init_scale", "growth_factor", "backoff_factor", "growth_interval"),
        "train": (
            "mode", "batch_size", "epochs", "activation_checkpointing", "lr", "weight_decay", "adamw_betas",
            "adamw_eps", "adamw_amsgrad", "grad_clip_norm", "display_iter",
            "snapshots_folder", "load_pretrain", "resume_optimizers", "resume_mode", "pretrain_dir",
        ),
        "scheduler": ("step_size", "gamma"),
        "loss": (
            "patch_size", "brightness", "smooth_a_weight", "smooth_b_weight",
            "exposure_weight", "local_color_weight", "global_color_weight",
            "denoising_ssim_weight", "denoising_gradient_weight",
            "denoising_smooth_weight", "denoising_scale", "ssim_data_range",
        ),
        "inference": ("checkpoint", "input", "output", "gt", "stage", "benchmark", "warmup"),
        "validation": ("enabled", "lpips_net"),
    }

    @classmethod
    def validate_config(cls, config: Mapping) -> None:
        """尽早报告缺失或非法配置，避免使用隐藏的训练参数默认值。"""
        if not isinstance(config, Mapping):
            raise ValueError("配置必须是 YAML 映射")
        missing = [key for key in ("bit_mode", "enhancement_guidance", "enhancement_backbone")
                   if key not in config]
        for section, fields in cls.REQUIRED_FIELDS.items():
            values = config.get(section)
            if not isinstance(values, Mapping):
                raise ValueError(f"配置节 {section} 必须是映射")
            missing.extend(f"{section}.{key}" for key in fields if key not in values)
        if missing:
            raise ValueError("配置缺少必填参数：" + ", ".join(missing))
        validate_bit_mode(config["bit_mode"])
        validate_enhancement_guidance(config["enhancement_guidance"])
        validate_enhancement_backbone(config["enhancement_backbone"])
        if config["train"]["mode"] not in cls.MODES:
            raise ValueError("train.mode 必须是 enhancement、denoising 或 joint")
        if config["train"]["resume_mode"] not in ("full", "weights", "legacy"):
            raise ValueError("train.resume_mode 必须是 full、weights 或 legacy")
        if config["inference"]["stage"] not in ("enhancement", "denoising"):
            raise ValueError("inference.stage 必须是 enhancement 或 denoising")
        if config["validation"]["lpips_net"] not in ("alex", "vgg", "squeeze"):
            raise ValueError("validation.lpips_net 必须是 alex、vgg 或 squeeze")
        if config["amp"]["dtype"] not in ("float16", "bfloat16"):
            raise ValueError("amp.dtype 必须是 float16 或 bfloat16")
        seed = config["runtime"]["seed"]
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63):
            raise ValueError("runtime.seed 必须是非负整数或显式历史配置的 null")

        # 仅检查配置值，不调整输入图像、损失公式或训练中的浮点运算。
        for section, key, minimum in (
            ("train", "batch_size", 1), ("train", "epochs", 1),
            ("train", "display_iter", 0), ("data", "image_size", 4),
            ("data", "num_workers", 0), ("scheduler", "step_size", 1),
            ("loss", "patch_size", 1),
            ("inference", "warmup", 0),
            ("amp", "growth_interval", 1),
        ):
            value = config[section][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{section}.{key} 必须是大于等于 {minimum} 的整数")
        # 原局部颜色损失使用固定的 4×4 池化，曝光池化不能大于训练图像。
        if config["loss"]["patch_size"] > config["data"]["image_size"]:
            raise ValueError("loss.patch_size 不能大于 data.image_size")
        for section, key in (
            ("train", "load_pretrain"), ("train", "resume_optimizers"),
            ("train", "activation_checkpointing"),
            ("train", "adamw_amsgrad"), ("data", "shuffle"),
            ("data", "pin_memory"), ("data", "drop_last"),
            ("data", "legacy_order"), ("runtime", "deterministic"), ("inference", "benchmark"),
            ("validation", "enabled"),
            ("amp", "enabled"),
        ):
            if not isinstance(config[section][key], bool):
                raise ValueError(f"{section}.{key} 必须是布尔值")
        for section, key in (
            ("train", "lr"), ("train", "adamw_eps"), ("scheduler", "gamma"),
            ("loss", "ssim_data_range"),
            ("amp", "init_scale"), ("amp", "growth_factor"), ("amp", "backoff_factor"),
        ):
            cls._check_number(config[section][key], f"{section}.{key}", positive=True)
        if config["amp"]["growth_factor"] <= 1:
            raise ValueError("amp.growth_factor 必须大于 1")
        if config["amp"]["backoff_factor"] >= 1:
            raise ValueError("amp.backoff_factor 必须小于 1")
        for key in ("weight_decay", "grad_clip_norm"):
            cls._check_number(config["train"][key], f"train.{key}")
        for key in cls.REQUIRED_FIELDS["loss"]:
            if key not in ("patch_size", "ssim_data_range"):
                cls._check_number(config["loss"][key], f"loss.{key}")
        betas = config["train"]["adamw_betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError("train.adamw_betas 必须包含两个数值")
        for beta in betas:
            cls._check_number(beta, "train.adamw_betas")
            if beta >= 1:
                raise ValueError("train.adamw_betas 中的数值必须小于 1")
        if not config["data"]["root"] or not config["train"]["snapshots_folder"]:
            raise ValueError("data.root 和 train.snapshots_folder 不能为空")
        if config["train"]["load_pretrain"] and not config["train"]["pretrain_dir"]:
            raise ValueError("开启 load_pretrain 时必须配置 train.pretrain_dir")
        validation = config["validation"]
        if "datasets" in validation:
            datasets = validation["datasets"]
            if not isinstance(datasets, Mapping) or not datasets:
                raise ValueError("validation.datasets 必须是非空的名称到目录配置映射")
            names = set()
            for name, values in datasets.items():
                if (not isinstance(name, str) or not name.strip() or name.strip() != name or name in (".", "..")
                        or "/" in name or "\\" in name or Path(name).is_absolute()
                        or any(ord(char) < 32 for char in name)):
                    raise ValueError("验证数据集名称必须是合法的单级文件夹名称")
                if name.casefold() in names:
                    raise ValueError("验证数据集名称不能重复（忽略大小写）")
                names.add(name.casefold())
                if not isinstance(values, Mapping) or any(not values.get(key) for key in ("low_dir", "high_dir")):
                    raise ValueError(f"validation.datasets.{name} 必须配置 low_dir 和 high_dir")
        elif validation["enabled"] and any(not validation.get(key) for key in ("low_dir", "high_dir")):
            raise ValueError("启用验证时必须配置 validation.datasets 或 validation.low_dir 和 validation.high_dir")
        if "output" in config and (not isinstance(config["output"], Mapping) or not config["output"].get("root")):
            raise ValueError("output.root 不能为空")

    @staticmethod
    def _check_number(value, name: str, positive: bool = False) -> None:
        """检查有限的非负数值；需要时进一步要求大于零。"""
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or (positive and value == 0)):
            requirement = "正数" if positive else "非负数"
            raise ValueError(f"{name} 必须是有限的{requirement}")

    def __init__(
        self,
        config: Mapping,
        device: Optional[Union[str, torch.device]] = None,
        *,
        bitplane_extractor=None,
        enhancement_network=None,
    ) -> None:
        self.validate_config(config)
        self.config = normalize_training_paths(deepcopy(config))
        config = self.config
        self.train_config = config["train"]
        self.data_config = config["data"]
        self.mode = self.train_config["mode"]
        # 显式的旧单验证集配置继续使用平面状态和 v4 检查点。
        self.multi_validation = config["validation"]["enabled"] and "datasets" in config["validation"]
        self.validation_names = tuple(config["validation"]["datasets"]) if self.multi_validation else ()
        self.train_enhancement = self.mode in ("enhancement", "joint")
        self.train_denoising = self.mode in ("denoising", "joint")
        self.device = torch.device(device if device is not None else config["runtime"]["device"])
        self.config["runtime"]["device"] = str(self.device)
        self.amp_enabled = config["amp"]["enabled"] and self.device.type == "cuda"
        self.amp_dtype_name = config["amp"]["dtype"]
        self.amp_dtype = getattr(torch, self.amp_dtype_name)
        if self.amp_enabled and self.amp_dtype == torch.bfloat16:
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("当前 CUDA 设备不支持 BF16，请选择 amp.dtype=float16")
        self._reset_amp_state()
        seed_everything(config["runtime"]["seed"], config["runtime"]["deterministic"])
        self.bit_mode = config["bit_mode"]
        self.enhancement_guidance = config["enhancement_guidance"]
        self.enhancement_backbone = config["enhancement_backbone"]
        self.model = SIMI(
            bit_mode=self.bit_mode, enhancement_guidance=self.enhancement_guidance,
            enhancement_backbone=self.enhancement_backbone,
            activation_checkpointing=self.train_config["activation_checkpointing"],
            bitplane_extractor=bitplane_extractor, enhancement_network=enhancement_network,
        )
        # 此轮只预留研究接口，不实现可学习提取器的优化、冻结与检查点策略。
        if any(True for _ in self.model.bitplane_extractor.parameters()):
            raise NotImplementedError("当前训练器尚未实现带参数位平面提取器的训练策略")
        self.model.to(self.device)
        self.model.light_net.requires_grad_(self.train_enhancement)
        self.model.denoise.requires_grad_(self.train_denoising)
        self.loss = UnsupervisedLoss(**config["loss"]).to(self.device)
        optimizer_options = {
            "lr": self.train_config["lr"],
            "weight_decay": self.train_config["weight_decay"],
            "betas": tuple(self.train_config["adamw_betas"]),
            "eps": self.train_config["adamw_eps"],
            "amsgrad": self.train_config["adamw_amsgrad"],
        }
        self.optimizer_light = torch.optim.AdamW(self.model.light_net.parameters(), **optimizer_options)
        self.optimizer_denoise = torch.optim.AdamW(self.model.denoise.parameters(), **optimizer_options)
        self.iteration = 0
        self.current_epoch = 0
        self.batch_in_epoch = 0
        self.dataset = None
        self.batch_sampler = None
        self.loader = None
        self._pending_data_state = None
        self._pending_rng_state = None
        self.last_metrics = {}
        self.validator = None
        self._reset_selection_state()
        self.timing_state = {"train_seconds": 0.0, "train_steps": 0, "epoch_seconds": 0.0, "epoch_count": 0}
        self.loader_generator = torch.Generator()
        self.worker_seed = torch.initial_seed() if config["runtime"]["seed"] is None else config["runtime"]["seed"]
        self.loader_generator.manual_seed(self.worker_seed)
        self.grad_clip_norm = self.train_config["grad_clip_norm"]

        scheduler_options = {
            "step_size": config["scheduler"]["step_size"],
            "gamma": config["scheduler"]["gamma"],
        }
        self.scheduler_light = torch.optim.lr_scheduler.StepLR(self.optimizer_light, **scheduler_options)
        self.scheduler_denoise = torch.optim.lr_scheduler.StepLR(self.optimizer_denoise, **scheduler_options)
        self._set_training_modes()
        # 先构造优化器和调度器，再完整恢复其状态，最后恢复随机状态。
        if self.train_config["load_pretrain"]:
            self.load_checkpoint(self.train_config["pretrain_dir"])

    @property
    def bitplane_extractor(self):
        """引用模型拥有的提取器，不重复创建或注册研究模块。"""
        return self.model.bitplane_extractor

    def _set_training_modes(self) -> None:
        """将活动分支设为训练模式，冻结分支设为评估模式。"""
        self.model.train()
        self.model.light_net.train(self.train_enhancement)
        self.model.denoise.train(self.train_denoising)
        self.model.bitplane_extractor.train(self.train_enhancement)

    def _reset_amp_state(self) -> None:
        """初始化当前精度的缩放器；仅加载权重时不继承旧实验缩放状态。"""
        options = {key: self.config["amp"][key] for key in (
            "init_scale", "growth_factor", "backoff_factor", "growth_interval",
        )}
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16, **options,
        )
        self.last_step_updates = {"light": False, "denoise": False}
        self.amp_skipped_steps = {"light": 0, "denoise": 0}

    def _update_branch(self, total: torch.Tensor, optimizer, parameters) -> bool:
        """缩放反向后先还原梯度，再裁剪；返回优化器是否实际完成更新。"""
        optimizer.zero_grad()
        self.scaler.scale(total).backward()
        self.scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm)
        # BF16 无需缩放器，但仍不能让非有限梯度污染模型与优化器。
        if self.amp_enabled and not self.scaler.is_enabled() and not bool(torch.isfinite(norm)):
            return False
        updated = False

        def mark_updated(*args):
            nonlocal updated
            updated = True

        # AdamW.step 成功时也返回 None，使用公开 hook 区分 GradScaler 的跳过行为。
        handle = optimizer.register_step_post_hook(mark_updated)
        try:
            self.scaler.step(optimizer)
        finally:
            handle.remove()
        return updated

    def train_step(self, rgb: torch.Tensor, planes: Optional[torch.Tensor] = None,
                   extra_guidance=None) -> dict:
        """执行当前模式的一次更新，返回参与训练的分支损失。

        默认输入仅含 RGB，在训练设备上批量提取位平面；显式 planes
        保留旧调用接口，适用于已从原始 uint8 像素提取位平面的数据。
        extra_guidance 仅透传给研究者提供的增强网络；默认实现不使用引导图，
        不会计算 SNR 或可靠性，非空值仍由默认增强网络明确拒绝。
        """
        rgb = rgb.to(device=self.device, dtype=torch.float32)
        if planes is None:
            planes = self.bitplane_extractor(rgb)
        else:
            planes = planes.to(device=self.device, dtype=torch.float32)
        if isinstance(extra_guidance, torch.Tensor):
            extra_guidance = extra_guidance.to(device=self.device, dtype=rgb.dtype)

        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            if self.mode == "joint":
                # 保持原始完整前向；去噪的输入由 SIMI 内部 detach。
                enhanced, curve_a, curve_b, final = self.model(rgb, planes, extra_guidance=extra_guidance)
            elif self.mode == "enhancement":
                enhanced, curve_a, curve_b, _ = self.model.light_net(rgb, planes, extra_guidance=extra_guidance)
            else:
                # 增强网络仅提供固定目标和特征，不构建其反向传播图。
                with torch.no_grad():
                    enhanced, _, _, features = self.model.light_net(rgb, planes, extra_guidance=extra_guidance)
                final = self.model.denoise(enhanced.detach(), features.detach())

        metrics = {}
        updates = {"light": False, "denoise": False}
        if self.train_enhancement:
            # 曲线平滑、颜色比例、曝光和大权重归约继续按原公式使用 FP32。
            with torch.autocast(self.device.type, enabled=False):
                light_losses = self.loss.enhancement(enhanced.float(), rgb, curve_a.float(), curve_b.float())
            updates["light"] = self._update_branch(light_losses["total"], self.optimizer_light, self.model.light_net.parameters())
            metrics.update({
                "loss_light": light_losses["total"],
                "loss_smo": light_losses["smoothness"],
                "loss_exp": light_losses["exposure"],
                "loss_locl": light_losses["local_color"],
                "loss_gcol": light_losses["global_color"],
            })
        if self.train_denoising:
            # joint 使用增强参数更新前的这次前向结果，不重新计算目标。
            with torch.autocast(self.device.type, enabled=False):
                noise_losses = self.loss.denoising(final.float(), enhanced.detach().float())
            updates["denoise"] = self._update_branch(noise_losses["total"], self.optimizer_denoise, self.model.denoise.parameters())
            metrics.update({
                "loss_noise": noise_losses["total"],
                "denoising_ssim": noise_losses["ssim"],
                "denoising_gradient": noise_losses["gradient"],
                "denoising_smoothness": noise_losses["smoothness"],
            })
        # 两个优化器各自检查溢出，同一缩放器只在本次所有分支处理完成后更新一次。
        self.scaler.update()
        self.last_step_updates = updates
        for name, active in (("light", self.train_enhancement), ("denoise", self.train_denoising)):
            if active and not updates[name]:
                self.amp_skipped_steps[name] += 1
        if updates["light"]:
            self.scheduler_light.step()
        if updates["denoise"]:
            self.scheduler_denoise.step()
        self.iteration += 1
        self.last_metrics = {name: value.detach() for name, value in metrics.items()}
        return self.last_metrics

    def _resume_signature(self) -> dict:
        """完整续训必须匹配数值设置；目标总轮数、日志和输出位置可调整。"""
        return {
            "model": self.model.checkpoint_config(),
            "mode": self.mode,
            "loss": deepcopy(self.config["loss"]),
            "optimizer": {key: self.train_config[key] for key in (
                "lr", "weight_decay", "adamw_betas", "adamw_eps", "adamw_amsgrad", "grad_clip_norm",
            )},
            "scheduler": deepcopy(self.config["scheduler"]),
            "data": {key: self.data_config[key] for key in (
                "image_size", "num_workers", "shuffle", "drop_last", "legacy_order",
            )},
            "batch_size": self.train_config["batch_size"],
            "activation_checkpointing": self.train_config["activation_checkpointing"],
            "seed": self.config["runtime"]["seed"],
            "deterministic": self.config["runtime"]["deterministic"],
            "device": str(self.device),
            "torch_version": str(torch.__version__),
            "amp": self._amp_signature(),
            "validation": self._validation_config_signature(),
            "environment": {
                "pillow": pillow_version, "threads": torch.get_num_threads(),
                "dtype": str(next(self.model.parameters()).dtype),
                "cudnn": torch.backends.cudnn.version(),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            },
        }

    def _validation_config_signature(self) -> dict:
        """指标协议和数据集名称影响最佳历史；绝对目录通过文件指纹另行验证。"""
        signature = {key: self.config["validation"][key] for key in ("enabled", "lpips_net")}
        if self.multi_validation:
            signature["datasets"] = list(self.validation_names)
        return signature

    def _amp_signature(self) -> dict:
        """记录真正生效的精度；CPU 和显式关闭 AMP 都对应原 FP32。"""
        if not self.amp_enabled:
            return {"enabled": False}
        return {
            "enabled": True, "dtype": self.amp_dtype_name,
            "scaler": {key: self.config["amp"][key] for key in (
                "init_scale", "growth_factor", "backoff_factor", "growth_interval",
            )} if self.scaler.is_enabled() else {},
        }

    def _signature_for_checkpoint(self, version: int) -> dict:
        """兼容没有 AMP 配置的 v2/v3 FP32 检查点，允许保留进度切换精度。"""
        signature = self._resume_signature()
        if version < 4:
            signature.pop("amp")
        if version < 5:
            signature["validation"].pop("datasets", None)
        if version == 2:
            signature.pop("validation")
        return signature

    def load_checkpoint(self, path: Union[str, Path], resume_optimizers: Optional[bool] = None,
                        *, resume_mode: Optional[str] = None) -> None:
        """full 恢复进度与随机状态；weights 新阶段加载；legacy 显式旧格式热启动。"""
        policy = self.train_config["resume_mode"] if resume_mode is None else resume_mode
        if resume_mode is None:
            restore = self.train_config["resume_optimizers"] if resume_optimizers is None else resume_optimizers
            if not restore:
                policy = "weights"
            elif policy == "weights" and resume_optimizers is True:
                policy = "legacy"
        if policy not in ("full", "weights", "legacy"):
            raise ValueError("resume_mode 必须是 full、weights 或 legacy")
        path = normalize_path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.validate_checkpoint_config(checkpoint)
        if policy == "full":
            fields = {"training_state", "training_config", "scheduler_light", "scheduler_denoise",
                      "optimizer_light", "optimizer_denoise"}
            checkpoint_version = checkpoint.get("checkpoint_version")
            if checkpoint_version not in (2, 3, 4, 5) or not fields.issubset(checkpoint):
                raise ValueError("此旧检查点没有完整训练状态；请显式选择 resume_mode=weights 或 legacy")
            if checkpoint_version >= 3 and not {"selection_state", "timing_state"}.issubset(checkpoint):
                raise ValueError("检查点缺少最佳指标或计时状态")
            if checkpoint_version >= 4 and not {"amp_scaler", "amp_skipped_steps"}.issubset(checkpoint):
                raise ValueError("AMP 检查点缺少缩放器或溢出计数状态")
            if checkpoint_version == 5 and (not self.multi_validation or "best_records" not in checkpoint["selection_state"]):
                raise ValueError("多数据集检查点需要相同的 validation.datasets 和完整最佳模型记录")
            if checkpoint["training_config"] != self._signature_for_checkpoint(checkpoint_version):
                raise ValueError("完整续训配置不匹配（AMP、loss、训练模式、设备或采样设置）；新实验请使用 resume_mode=weights")
        state = checkpoint["net"] if "net" in checkpoint else checkpoint
        self.model.load_state_dict(state, strict=True)
        if policy in ("full", "legacy"):
            if "optimizer_light" not in checkpoint or "optimizer_denoise" not in checkpoint:
                raise ValueError("恢复优化器需要包含两个优化器状态的训练检查点；裸权重请使用 resume_mode=weights")
            self.optimizer_light.load_state_dict(checkpoint["optimizer_light"])
            self.optimizer_denoise.load_state_dict(checkpoint["optimizer_denoise"])
        if policy == "full":
            self.scheduler_light.load_state_dict(checkpoint["scheduler_light"])
            self.scheduler_denoise.load_state_dict(checkpoint["scheduler_denoise"])
            progress = checkpoint["training_state"]
            self.current_epoch = progress["epoch"]
            self.batch_in_epoch = progress["batch_in_epoch"]
            self.iteration = progress["global_step"]
            self.worker_seed = progress["worker_seed"]
            self.loader_generator.set_state(progress["loader_generator"].cpu())
            self._pending_data_state = progress
            self._pending_rng_state = progress["rng"]
            self._reset_selection_state()
            if checkpoint_version >= 3:
                selection = checkpoint["selection_state"]
                if self.multi_validation and checkpoint_version < 5:
                    # 先保存旧历史，实际构造六个验证集后按唯一文件指纹迁移。
                    self._legacy_selection_pending = deepcopy(selection)
                else:
                    self.best_scores = deepcopy(selection["best_scores"])
                    self.best_epochs = deepcopy(selection["best_epochs"])
                    self.best_records = deepcopy(selection.get("best_records", self.best_records))
                    self.last_validation = deepcopy(selection["last_validation"])
                    self.validation_history = deepcopy(selection["validation_history"])
                    self._validation_dataset_signature = deepcopy(selection["dataset_signature"])
                self.timing_state = deepcopy(checkpoint["timing_state"])
                self._resume_checkpoint_path = Path(path)
            else:
                print("已恢复 v2 训练进度；旧检查点没有验证历史，将从下一次完整 epoch 验证建立最佳指标。", flush=True)
                self.timing_state = {"train_seconds": 0.0, "train_steps": 0, "epoch_seconds": 0.0, "epoch_count": 0}
            self._reset_amp_state()
            if checkpoint_version >= 4:
                self.scaler.load_state_dict(checkpoint["amp_scaler"])
                self.amp_skipped_steps = deepcopy(checkpoint["amp_skipped_steps"])
            elif self.amp_enabled:
                print("已恢复 FP32 检查点的训练进度；接下来采用 AMP，梯度缩放器从当前配置初始化。", flush=True)
            self.validator = None
            restore_rng_state(self._pending_rng_state)
            if self.loader is not None:
                self.loader = None
                self.dataset = None
                self.batch_sampler = None
        else:
            # 新阶段和旧式热启动都从新的进度开始，不伪装成精确续训。
            self.current_epoch = self.batch_in_epoch = self.iteration = 0
            self._pending_data_state = self._pending_rng_state = None
            self.loader = self.dataset = self.batch_sampler = None
            self.validator = None
            self._reset_selection_state()
            self.timing_state = {"train_seconds": 0.0, "train_steps": 0, "epoch_seconds": 0.0, "epoch_count": 0}
            self._reset_amp_state()
            if policy == "weights":
                # 已使用过的 Trainer 也重新创建优化器，避免残留上一阶段动量。
                options = {
                    "lr": self.train_config["lr"], "weight_decay": self.train_config["weight_decay"],
                    "betas": tuple(self.train_config["adamw_betas"]), "eps": self.train_config["adamw_eps"],
                    "amsgrad": self.train_config["adamw_amsgrad"],
                }
                self.optimizer_light = torch.optim.AdamW(self.model.light_net.parameters(), **options)
                self.optimizer_denoise = torch.optim.AdamW(self.model.denoise.parameters(), **options)
            options = {"step_size": self.config["scheduler"]["step_size"], "gamma": self.config["scheduler"]["gamma"]}
            self.scheduler_light = torch.optim.lr_scheduler.StepLR(self.optimizer_light, **options)
            self.scheduler_denoise = torch.optim.lr_scheduler.StepLR(self.optimizer_denoise, **options)
        self._set_training_modes()

    def _dataset_state(self) -> dict:
        root = Path(self.data_config["root"])
        paths = [str(Path(path).relative_to(root)) for path in self.dataset.data_list]
        entries = []
        for relative in paths:
            properties = (root / relative).stat()
            entries.append((relative, properties.st_size, properties.st_mtime_ns))
        fingerprint = hashlib.sha256(repr(sorted(entries)).encode()).hexdigest()
        return {"paths": paths, "fingerprint": fingerprint}

    def _reset_selection_state(self) -> None:
        """新阶段独立选择最佳权重；完整续训会覆盖为保存的历史。"""
        empty = {name: None for name in ("psnr", "ssim", "lpips")}
        self.best_scores = {name: deepcopy(empty) for name in self.validation_names} if self.multi_validation else deepcopy(empty)
        self.best_epochs = deepcopy(self.best_scores)
        self.best_records = deepcopy(self.best_scores)
        self.last_validation = {}
        self.validation_history = []
        self._validation_dataset_signature = None
        self._improved_metrics = []
        self._resume_checkpoint_path = None
        self._legacy_selection_pending = None

    def _prepare_validator(self) -> None:
        """验证开始前校验数据及评价器，避免训练一轮后才发现路径错误。"""
        if not self.config["validation"]["enabled"]:
            return
        if self.validator is None:
            # 评价器构造不能消耗后续训练所需的随机状态。
            rng_state = capture_rng_state()
            try:
                validator_class = MultiDatasetValidator if self.multi_validation else EpochValidator
                self.validator = validator_class(self.config["validation"], self.device)
            finally:
                restore_rng_state(rng_state)
        signature = self.validator.signature()
        if self.multi_validation and set(signature) != set(self.validation_names):
            raise ValueError("评价器的数据集名称与 validation.datasets 不一致")
        if self._legacy_selection_pending is not None:
            self._migrate_legacy_selection(signature)
        if (self._validation_dataset_signature is not None
                and self._validation_dataset_signature != signature):
            raise ValueError("完整续训的验证集或指标定义已变化；新实验请使用 resume_mode=weights")
        self._validation_dataset_signature = signature

    def _migrate_legacy_selection(self, signature: dict) -> None:
        """仅将旧单集历史交给文件与指标指纹相同的唯一数据集，其余集从首次评价开始。"""
        selection = self._legacy_selection_pending
        old_signature = selection["dataset_signature"]
        has_best = any(value is not None for value in selection["best_scores"].values())
        if old_signature is None and not has_best:
            self._legacy_selection_pending = None
            return
        names = [name for name, value in signature.items() if value == old_signature]
        if len(names) != 1:
            raise ValueError("旧单验证集指纹无法唯一匹配当前测试集；请保留原数据，或以 resume_mode=weights 开始新实验")
        dataset = names[0]
        records = {metric: None for metric in ("psnr", "ssim", "lpips")}
        for metric, score in selection["best_scores"].items():
            if score is None:
                continue
            epoch = selection["best_epochs"][metric]
            candidates = [record for record in selection["validation_history"]
                          if record["epoch"] == epoch and record[metric] == score]
            if len(candidates) != 1:
                raise ValueError("旧验证历史缺少最佳模型当轮的完整指标，无法生成可靠记录")
            records[metric] = deepcopy(candidates[0])
        self.best_scores[dataset] = deepcopy(selection["best_scores"])
        self.best_epochs[dataset] = deepcopy(selection["best_epochs"])
        self.best_records[dataset] = records
        if selection["last_validation"]:
            self.last_validation = {dataset: deepcopy(selection["last_validation"])}
        self.validation_history = [
            {"epoch": record["epoch"], "global_step": record["global_step"],
             "datasets": {dataset: {key: record[key] for key in ("psnr", "ssim", "lpips", "num_images")}}}
            for record in selection["validation_history"]
        ]
        self._legacy_selection_pending = None
        print(f"已将旧单验证集的最佳权重历史迁移至 {dataset}，其余测试集将在下一轮首次评价。", flush=True)

    def validate_epoch(self, epoch: int) -> dict:
        """每轮评价一次，独立记录 PSNR/SSIM 最大值与 LPIPS 最小值。"""
        self._improved_metrics = []
        if not self.config["validation"]["enabled"]:
            return {}
        self._prepare_validator()
        rng_state = capture_rng_state()
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            with torch.inference_mode(), torch.autocast(self.device.type, enabled=False):
                scores = self.validator.validate(self.model, self.mode)
        finally:
            for module, training in modes:
                module.training = training
            restore_rng_state(rng_state)
        if self.multi_validation:
            if not isinstance(scores, Mapping) or set(scores) != set(self.validation_names):
                raise ValueError("验证结果必须包含全部配置的数据集")
            # 全部指标先校验，再整体提交；任一数据集失败不能推进部分最佳历史。
            results = {name: self._checked_validation_scores(scores[name]) for name in self.validation_names}
            for dataset, values in results.items():
                for metric in ("psnr", "ssim", "lpips"):
                    old, new = self.best_scores[dataset][metric], values[metric]
                    if old is None or (new < old if metric == "lpips" else new > old):
                        self.best_scores[dataset][metric] = new
                        self.best_epochs[dataset][metric] = epoch
                        self.best_records[dataset][metric] = {"epoch": epoch, "global_step": self.iteration, **values}
                        self._improved_metrics.append((dataset, metric))
            self.last_validation = results
            self.validation_history.append({"epoch": epoch, "global_step": self.iteration, "datasets": deepcopy(results)})
            return results
        scores = self._checked_validation_scores(scores)
        for name in self.best_scores:
            old, new = self.best_scores[name], scores[name]
            improved = old is None or (new < old if name == "lpips" else new > old)
            if improved:
                self.best_scores[name] = new
                self.best_epochs[name] = epoch
                self.best_records[name] = {"epoch": epoch, "global_step": self.iteration, **scores}
                self._improved_metrics.append(name)
        self.last_validation = scores
        self.validation_history.append({"epoch": epoch, "global_step": self.iteration, **scores})
        return scores

    @staticmethod
    def _checked_validation_scores(scores: Mapping) -> dict:
        """允许完美重建时的正无穷 PSNR，拒绝空集及其它非有限指标。"""
        count = scores["num_images"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("验证指标必须来自非空配对集")
        scores = {**{name: float(scores[name]) for name in ("psnr", "ssim", "lpips")}, "num_images": count}
        if (math.isnan(scores["psnr"]) or scores["psnr"] == -math.inf
                or not all(math.isfinite(scores[name]) for name in ("ssim", "lpips"))):
            raise ValueError("验证指标包含非法数值，不能保存最佳权重")
        return scores

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """将非负秒数显示为天、小时、分钟和秒。"""
        seconds = max(0, int(seconds))
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"

    def _eta_seconds(self, pending_epoch: bool = False) -> float:
        """按已观测训练及轮末验证耗时估计剩余时间；暂停期间不计时。"""
        timing = self.timing_state
        total_steps = self.train_config["epochs"] * self.batch_sampler.total_batches
        remaining_steps = max(0, total_steps - self.iteration)
        mean_step = timing["train_seconds"] / timing["train_steps"] if timing["train_steps"] else 0.0
        mean_epoch = timing["epoch_seconds"] / timing["epoch_count"] if timing["epoch_count"] else 0.0
        remaining_epochs = max(0, self.train_config["epochs"] - self.current_epoch) + int(pending_epoch)
        return remaining_steps * mean_step + remaining_epochs * mean_epoch

    def _log_progress(self, epoch: int, batch: int, *, pending_epoch: bool = False) -> None:
        """打印全局进度、当前学习率、当前批次 loss 和预计剩余训练时间。"""
        total_steps = self.train_config["epochs"] * self.batch_sampler.total_batches
        values = ", ".join(f"{name}: {value.item():.4f}" for name, value in self.last_metrics.items())
        eta = self._format_duration(self._eta_seconds(pending_epoch))
        print(
            f"[{self.mode}] Epoch {epoch + 1}/{self.train_config['epochs']} "
            f"Batch {batch}/{self.batch_sampler.total_batches} Iter {self.iteration}/{total_steps} "
            f"lr_light={self.optimizer_light.param_groups[0]['lr']:.6g} "
            f"lr_denoise={self.optimizer_denoise.param_groups[0]['lr']:.6g} "
            f"{values} precision={self.amp_dtype_name if self.amp_enabled else 'float32'} "
            f"loss_scale={self.scaler.get_scale():g} "
            f"skipped_light={self.amp_skipped_steps['light']} skipped_denoise={self.amp_skipped_steps['denoise']} "
            f"ETA={eta}", flush=True,
        )

    def _save_epoch_checkpoint(self, epoch: int, snapshots_folder: Path) -> None:
        """完成轮末验证后统一保存，best 与 latest 使用相同的完整训练状态。"""
        started = time.perf_counter()
        scores = self.validate_epoch(epoch)
        self.timing_state["epoch_seconds"] += time.perf_counter() - started
        self.timing_state["epoch_count"] += 1
        if self.multi_validation:
            for dataset, metric in self._improved_metrics:
                self.save_checkpoint(snapshots_folder / dataset / f"best_{metric}.pth")
        else:
            for name in self._improved_metrics:
                self.save_checkpoint(snapshots_folder / f"best_{name}.pth")
        self.save_checkpoint(snapshots_folder / "latest.pth")
        if self.multi_validation:
            self._write_best_summaries(snapshots_folder)
            self._write_validation_history()
            for dataset, values in scores.items():
                updated = ",".join(metric for name, metric in self._improved_metrics if name == dataset) or "none"
                print(
                    f"[validation:{dataset}] Epoch {epoch + 1}/{self.train_config['epochs']} "
                    f"images={values['num_images']} PSNR={values['psnr']:.4f} "
                    f"SSIM={values['ssim']:.6f} LPIPS={values['lpips']:.6f} "
                    f"best_updated={updated} ETA={self._format_duration(self._eta_seconds())}", flush=True,
                )
            return
        if scores:
            updated = ",".join(self._improved_metrics) or "none"
            print(
                f"[validation] Epoch {epoch + 1}/{self.train_config['epochs']} "
                f"images={scores['num_images']} PSNR={scores['psnr']:.4f} "
                f"SSIM={scores['ssim']:.6f} LPIPS={scores['lpips']:.6f} "
                f"best_updated={updated} ETA={self._format_duration(self._eta_seconds())}", flush=True,
            )

    @staticmethod
    def _write_atomic_text(path: Path, content: str) -> None:
        """文本也先写临时文件再替换，防止读取到半份实时记录。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_best_summaries(self, snapshots_folder: Path) -> None:
        """每集分别记录三个最佳模型当轮的完整指标，展示的轮次从 1 开始。"""
        for dataset in self.validation_names:
            lines = ["=" * 64, "Best Models Metrics Summary", "=" * 64,
                     f"Dataset: {dataset}", "LPIPS: lower is better", ""]
            for metric in ("psnr", "ssim", "lpips"):
                lines.extend([f"Best {metric.upper()} Model", "-" * 64])
                record = self.best_records[dataset][metric]
                if record is None:
                    lines.extend(["Status: Not evaluated yet", ""])
                    continue
                lines.extend([
                    f"Model: best_{metric}.pth", f"Epoch: {record['epoch'] + 1}",
                    f"Global Step: {record['global_step']}", f"PSNR: {record['psnr']:.4f} dB",
                    f"SSIM: {record['ssim']:.4f}", f"LPIPS: {record['lpips']:.4f}",
                    f"Images: {record['num_images']}", "",
                ])
            self._write_atomic_text(snapshots_folder / dataset / "best_models_metrics.txt", "\n".join(lines) + "\n")

    def _write_validation_history(self) -> None:
        """由检查点中的历史重建 CSV，完整续训及换目录不会重复追加旧轮次。"""
        if not self.multi_validation or "output" not in self.config:
            return
        import io
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(("epoch", "global_step", "dataset", "num_images", "psnr", "ssim", "lpips"))
        for record in self.validation_history:
            for dataset, scores in record["datasets"].items():
                writer.writerow((record["epoch"] + 1, record["global_step"], dataset, scores["num_images"],
                                 scores["psnr"], scores["ssim"], scores["lpips"]))
        self._write_atomic_text(Path(self.config["output"]["root"]) / "logs" / "validation_metrics.csv", stream.getvalue())

    def _restore_best_files(self, snapshots_folder: Path) -> None:
        """校验恢复历史的最佳权重；换输出目录时迁移正确文件，不能以当前权重替代。"""
        if self._resume_checkpoint_path is None:
            return
        if self.multi_validation:
            self._restore_multi_best_files(snapshots_folder)
            return
        copies = []

        def matches(path: Path, name: str) -> bool:
            if not path.is_file():
                return False
            payload = torch.load(path, map_location="cpu", weights_only=True)
            selection = payload.get("selection_state", {})
            return (
                payload.get("model_config") == self.model.checkpoint_config()
                and payload.get("checkpoint_version") in (3, 4)
                and payload.get("training_config") == self._signature_for_checkpoint(payload["checkpoint_version"])
                and selection.get("dataset_signature") == self._validation_dataset_signature
                and selection.get("best_epochs", {}).get(name) == self.best_epochs[name]
                and selection.get("last_validation", {}).get(name) == self.best_scores[name]
                and payload.get("training_state", {}).get("epoch") == self.best_epochs[name] + 1
            )

        for name, score in self.best_scores.items():
            if score is None:
                continue
            destination = snapshots_folder / f"best_{name}.pth"
            if destination.exists():
                if not matches(destination, name):
                    raise ValueError(f"{destination.name} 与恢复的最佳指标历史不一致；请使用匹配的历史权重或以 weights 开始新实验")
                continue
            candidates = (self._resume_checkpoint_path.parent / destination.name, self._resume_checkpoint_path)
            source = next((candidate for candidate in candidates if matches(candidate, name)), None)
            if source is None:
                raise ValueError(f"缺少与恢复历史匹配的 {destination.name}；请提供历史最佳权重或以 weights 开始新实验")
            copies.append((source, destination))
        # 先验证所有历史文件，再执行迁移，避免发现缺失时留下部分新文件。
        for source, destination in copies:
            temporary = destination.with_name(destination.name + ".tmp")
            try:
                shutil.copyfile(source, temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        self._resume_checkpoint_path = None

    def _restore_multi_best_files(self, snapshots_folder: Path) -> None:
        """逐集验证真实历史最佳权重，可迁移旧单集文件；最新权重不能冒充历史最佳。"""
        copies = []

        def matches(path: Path, dataset: str, metric: str) -> bool:
            if not path.is_file():
                return False
            payload = torch.load(path, map_location="cpu", weights_only=True)
            version = payload.get("checkpoint_version")
            if (version not in (3, 4, 5) or payload.get("model_config") != self.model.checkpoint_config()
                    or payload.get("training_config") != self._signature_for_checkpoint(version)):
                return False
            selection = payload.get("selection_state", {})
            record = self.best_records[dataset][metric]
            if record is None:
                return False
            if version == 5:
                if selection.get("dataset_signature") != self._validation_dataset_signature:
                    return False
                epoch = selection.get("best_epochs", {}).get(dataset, {}).get(metric)
                scores = selection.get("last_validation", {}).get(dataset, {})
                if selection.get("best_records", {}).get(dataset, {}).get(metric) != record:
                    return False
            else:
                if selection.get("dataset_signature") != self._validation_dataset_signature[dataset]:
                    return False
                epoch = selection.get("best_epochs", {}).get(metric)
                scores = selection.get("last_validation", {})
            progress = payload.get("training_state", {})
            return (epoch == record["epoch"] and progress.get("epoch") == record["epoch"] + 1
                    and progress.get("global_step") == record["global_step"]
                    and all(scores.get(key) == record[key] for key in ("psnr", "ssim", "lpips", "num_images")))

        for dataset in self.validation_names:
            for metric, score in self.best_scores[dataset].items():
                if score is None:
                    continue
                destination = snapshots_folder / dataset / f"best_{metric}.pth"
                if destination.exists():
                    if not matches(destination, dataset, metric):
                        raise ValueError(f"{dataset}/{destination.name} 与恢复的最佳指标历史不一致；请使用匹配的历史权重或以 weights 开始新实验")
                    continue
                parent = self._resume_checkpoint_path.parent
                candidates = [parent / dataset / destination.name, parent / destination.name,
                              self._resume_checkpoint_path]
                if parent.name in self.validation_names:
                    candidates.append(parent.parent / dataset / destination.name)
                source = next((candidate for candidate in candidates if matches(candidate, dataset, metric)), None)
                if source is None:
                    raise ValueError(f"缺少与恢复历史匹配的 {dataset}/{destination.name}；请提供历史最佳权重或以 weights 开始新实验")
                copies.append((source, destination))
        # 六集全部确认通过后才复制，避免缺文件时留下部分新的最佳权重。
        for source, destination in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            try:
                shutil.copyfile(source, temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        self._write_best_summaries(snapshots_folder)
        self._write_validation_history()
        self._resume_checkpoint_path = None

    def build_loader(self) -> DataLoader:
        """构造独立采样器及工作进程种子，验证恢复时的数据清单。"""
        if self.loader is not None:
            return self.loader
        if (self._pending_data_state is not None
                and self._pending_data_state["dataset"] is None
                and self.iteration > 0):
            raise ValueError("此检查点由外部 train_step 保存，不含数据进度；请继续 train_step，或选择 resume_mode=weights 开始新训练")
        self.dataset = LowLightDataset(
            self.data_config["root"], image_size=self.data_config["image_size"], return_bitplanes=False,
            legacy_order=self.data_config["legacy_order"],
        )
        if not len(self.dataset):
            raise ValueError(f"data.root 中没有匹配的场景 JPG 图像：{self.data_config['root']}")
        self.batch_sampler = ResumableBatchSampler(
            len(self.dataset), self.train_config["batch_size"], self.data_config["shuffle"],
            self.data_config["drop_last"], self.config["runtime"]["seed"],
        )
        if self.batch_sampler.total_batches == 0:
            raise ValueError("当前 batch_size 和 drop_last 配置不会产生任何训练批次")
        if self._pending_data_state is not None and self._pending_data_state["dataset"] is not None:
            saved = self._pending_data_state["dataset"]
            if saved["fingerprint"] != self._dataset_state()["fingerprint"]:
                raise ValueError("完整续训的数据清单或文件属性已变化")
            root = Path(self.data_config["root"])
            self.dataset.data_list = [str(root / relative) for relative in saved["paths"]]
            self.dataset.train_list = self.dataset.data_list
            self.batch_sampler.load_state_dict(self._pending_data_state["sampler"])
            if (self.current_epoch != self.batch_sampler.epoch
                    or self.batch_in_epoch != self.batch_sampler.cursor):
                raise ValueError("检查点训练进度与 sampler 游标不一致")
        self.loader = DataLoader(
            self.dataset, batch_sampler=self.batch_sampler,
            num_workers=self.data_config["num_workers"], pin_memory=self.data_config["pin_memory"],
            generator=self.loader_generator, worker_init_fn=seed_worker,
        )
        # 重建历史文件列表可能使用 Python random，精确恢复在准备完数据后再执行。
        if self._pending_rng_state is not None:
            restore_rng_state(self._pending_rng_state)
        self._pending_data_state = self._pending_rng_state = None
        return self.loader

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """原子保存模型、优化器、调度器、已确认的数据位置和所有随机源。"""
        if self._legacy_selection_pending is not None:
            # API 调用者可能在 fit 前直接保存；先迁移旧历史，避免 v5 写入空最佳状态。
            self._prepare_validator()
        path = Path(normalize_path(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = self._pending_data_state
        sampler = self.batch_sampler.state_dict() if self.batch_sampler is not None else (pending["sampler"] if pending else None)
        dataset = self._dataset_state() if self.dataset is not None else (pending["dataset"] if pending else None)
        progress = {
            "epoch": self.current_epoch, "batch_in_epoch": self.batch_in_epoch,
            "global_step": self.iteration, "worker_seed": self.worker_seed,
            "sampler": sampler, "dataset": dataset,
            "loader_generator": self.loader_generator.get_state(), "rng": capture_rng_state(),
        }
        checkpoint = {
            "checkpoint_version": 5 if self.multi_validation else 4,
            "net": self.model.legacy_state_dict() if self.enhancement_backbone == "legacy_cnn" else self.model.state_dict(),
            "model_config": self.model.checkpoint_config(), "training_config": self._resume_signature(),
            "optimizer_light": self.optimizer_light.state_dict(),
            "optimizer_denoise": self.optimizer_denoise.state_dict(),
            "scheduler_light": self.scheduler_light.state_dict(),
            "scheduler_denoise": self.scheduler_denoise.state_dict(), "training_state": progress,
            "selection_state": {
                "best_scores": deepcopy(self.best_scores), "best_epochs": deepcopy(self.best_epochs),
                "best_records": deepcopy(self.best_records),
                "last_validation": deepcopy(self.last_validation),
                "validation_history": deepcopy(self.validation_history),
                "dataset_signature": deepcopy(self._validation_dataset_signature),
            },
            "timing_state": deepcopy(self.timing_state),
            "amp_scaler": self.scaler.state_dict(),
            "amp_skipped_steps": deepcopy(self.amp_skipped_steps),
        }
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def fit(self, max_steps: Optional[int] = None) -> dict:
        """每轮验证并保存三个最佳模型和 latest；指定停点也更新 latest。"""
        if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0):
            raise ValueError("max_steps 必须是非负整数或 None")
        loader = self.build_loader()
        confirmed_steps = self.batch_sampler.epoch * self.batch_sampler.total_batches + self.batch_sampler.cursor
        if self.iteration != confirmed_steps:
            raise ValueError("global step 与已确认的数据进度不一致；外部 train_step 更新不能直接作为 fit 的完整续训")
        snapshots_folder = Path(self.train_config["snapshots_folder"])
        snapshots_folder.mkdir(parents=True, exist_ok=True)
        display_iter = self.train_config["display_iter"]
        self._set_training_modes()
        self._prepare_validator()
        self._restore_best_files(snapshots_folder)
        while self.current_epoch < self.train_config["epochs"]:
            if self.batch_sampler.cursor == self.batch_sampler.total_batches:
                # 接受显式手动提交后保存的轮末游标，并规范化为下一轮。
                epoch = self.current_epoch
                self.batch_sampler.next_epoch()
                self.current_epoch = self.batch_sampler.epoch
                self.batch_in_epoch = 0
                self._save_epoch_checkpoint(epoch, snapshots_folder)
                continue
            if max_steps is not None and self.iteration >= max_steps:
                self.save_checkpoint(snapshots_folder / "latest.pth")
                return self.last_metrics
            # 同一 epoch 重建 iterator 时重放其基础种子，避免额外消耗 RNG。
            self.loader_generator.manual_seed((self.worker_seed + self.current_epoch) % (2 ** 63))
            started = time.perf_counter()
            for rgb in loader:
                self.train_step(rgb)
                # CUDA 更新异步执行，计时前同步以免 ETA 只统计提交运算的时间。
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self.timing_state["train_seconds"] += time.perf_counter() - started
                self.timing_state["train_steps"] += 1
                self.batch_sampler.commit()
                self.batch_in_epoch = self.batch_sampler.cursor
                batch = self.batch_in_epoch
                finished_epoch = self.batch_sampler.cursor == self.batch_sampler.total_batches
                epoch = self.current_epoch
                if finished_epoch:
                    self.batch_sampler.next_epoch()
                    self.current_epoch = self.batch_sampler.epoch
                    self.batch_in_epoch = 0
                if display_iter > 0 and self.iteration % display_iter == 0:
                    self._log_progress(epoch, batch, pending_epoch=finished_epoch)
                if finished_epoch:
                    self._save_epoch_checkpoint(epoch, snapshots_folder)
                if max_steps is not None and self.iteration >= max_steps:
                    if not finished_epoch:
                        self.save_checkpoint(snapshots_folder / "latest.pth")
                    return self.last_metrics
                if finished_epoch:
                    break
                started = time.perf_counter()
        self.save_checkpoint(snapshots_folder / "latest.pth")
        return self.last_metrics
