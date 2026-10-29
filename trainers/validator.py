"""每轮使用配对参考图像评价增强结果，评价器不参与零参考训练。"""

from collections.abc import Mapping
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path

import torch
from PIL import Image

from utils.config import normalize_path
from utils.image_utils import image_to_tensor, load_image, read_rgb_array
from utils.metrics import image_metrics
from utils.reproducibility import capture_rng_state, restore_rng_state


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _paired_image_paths(low_dir: Path, high_dir: Path) -> list[tuple[Path, Path]]:
    """集中检查同名配对，供单数据集和多数据集在构造指标前共同使用。"""
    if low_dir.resolve() == high_dir.resolve():
        raise ValueError("验证输入与参考图像不能使用同一个目录")
    named_images = []
    for directory in (low_dir, high_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"验证图像目录不存在：{directory}")
        images = {
            path.name: path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        if not images:
            raise ValueError(f"验证图像目录为空：{directory}")
        named_images.append(images)
    low, high = named_images
    if set(low) != set(high):
        missing_high = sorted(set(low) - set(high))
        missing_low = sorted(set(high) - set(low))
        raise ValueError(
            f"验证集必须逐文件同名配对：缺少 high={missing_high}，缺少 low={missing_low}"
        )
    pairs = [(low[name], high[name]) for name in sorted(low)]
    for source, target in pairs:
        if source.resolve() == target.resolve():
            raise ValueError(f"验证参考图像不能指向对应输入本身：{source.name}")
        with Image.open(source) as low_image, Image.open(target) as high_image:
            if low_image.size != high_image.size:
                raise ValueError(f"验证配对图像尺寸不一致：{source.name}")
    return pairs


def _create_lpips_model(net: str, device: torch.device) -> torch.nn.Module:
    """构造一份官方冻结评价器，并消除临时参数初始化对训练随机源的影响。"""
    rng_state = capture_rng_state()
    try:
        import lpips

        return lpips.LPIPS(
            net=net, version="0.1", pretrained=True,
            pnet_rand=False, pnet_tune=False, eval_mode=True, verbose=False,
        ).to(device).eval().requires_grad_(False)
    finally:
        restore_rng_state(rng_state)


class EpochValidator:
    """逐张验证 PSNR、SSIM 和官方 LPIPS，并对每张图的指标取平均。

    low_dir 与 high_dir 必须包含同名 RGB 配对图像。验证保持输入尺寸，
    直接评价模型的浮点输出，不截断、不量化，也不把参考图像送入训练损失。
    LPIPS 是独立冻结的评价器；其参数不属于 SIMI 或训练优化器。
    """

    def __init__(self, config: dict, device: torch.device | str, *,
                 lpips_model: torch.nn.Module | None = None):
        self.device = torch.device(device)
        self.low_dir = Path(normalize_path(config["low_dir"]))
        self.high_dir = Path(normalize_path(config["high_dir"]))
        self.lpips_net = config["lpips_net"]
        if self.lpips_net not in ("alex", "vgg", "squeeze"):
            raise ValueError("validation.lpips_net 必须是 alex、vgg 或 squeeze")
        self.pairs = self._build_pairs()
        self._signature = self._build_signature()

        # 多数据集入口注入同一个官方评价器；旧接口仍独立构造一份。
        if lpips_model is None:
            self.lpips_metric = _create_lpips_model(self.lpips_net, self.device)
        else:
            self.lpips_metric = lpips_model.to(self.device).eval().requires_grad_(False)

    def _build_pairs(self) -> list[tuple[Path, Path]]:
        """在训练开始前检查缺失配对、空目录和图像尺寸错误。"""
        return _paired_image_paths(self.low_dir, self.high_dir)

    def _build_signature(self) -> dict:
        """记录相对文件名、大小和修改时间；验证目录迁移不改变指纹。"""
        records = []
        for source, target in self.pairs:
            low_stat, high_stat = source.stat(), target.stat()
            with Image.open(source) as image:
                dimensions = list(image.size)
            records.append({
                "name": source.name, "dimensions": dimensions,
                "low": [low_stat.st_size, low_stat.st_mtime_ns],
                "high": [high_stat.st_size, high_stat.st_mtime_ns],
            })
        encoded = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return {
            "num_images": len(records), "pairs_fingerprint": sha256(encoded).hexdigest(),
            "metrics": {
                "psnr_data_range": 1.0, "ssim_data_range": 1.0,
                "ssim_library": "pytorch_msssim", "ssim_version": version("pytorch-msssim"),
                "lpips_net": self.lpips_net, "lpips_version": "0.1",
                "lpips_package_version": version("lpips"),
                "lpips_pretrained": True, "lpips_input_normalization": "2*image-1",
                "resize": False, "clamp": False, "average": "per_image",
            },
        }

    def signature(self) -> dict:
        """返回独立副本，供完整续训检查验证集和指标定义。"""
        return json.loads(json.dumps(self._signature))

    def validate(self, model: torch.nn.Module, mode: str) -> dict:
        """评价当前训练阶段输出，并恢复每个子模块的训练状态及全部随机源。"""
        if mode not in ("enhancement", "denoising", "joint"):
            raise ValueError("验证 mode 必须是 enhancement、denoising 或 joint")
        rng_state = capture_rng_state()
        training_states = [(module, module.training) for module in model.modules()]
        results = []
        try:
            model.eval()
            self.lpips_metric.eval()
            with torch.inference_mode():
                for source, target_path in self.pairs:
                    pixels = read_rgb_array(source)
                    rgb = image_to_tensor(pixels).unsqueeze(0).to(self.device)
                    # 与现有推理入口一致，从 uint8 RGB 提取精确二进制位平面。
                    integer_rgb = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    planes = model.bitplane_extractor(integer_rgb)
                    if mode == "enhancement":
                        output = model.light_net(rgb, planes)[0]
                    else:
                        output = model(rgb, planes)[3]
                    target = load_image(target_path).unsqueeze(0).to(self.device)
                    if output.shape != target.shape:
                        raise ValueError(f"模型输出与验证参考图像尺寸不一致：{source.name}")
                    if not torch.isfinite(output).all():
                        raise ValueError(f"模型输出包含非有限数值：{source.name}")
                    measured = image_metrics(output, target)
                    # 官方 LPIPS 接收 RGB [-1,+1]；保持模型浮点输出的线性归一化。
                    measured["lpips"] = self.lpips_metric(output * 2 - 1, target * 2 - 1).mean()
                    values = {name: float(value) for name, value in measured.items()}
                    # 完全相同图像的 PSNR=+inf 是合法结果，不能误判为评价失败。
                    if any(math.isnan(value) or value == -math.inf for value in values.values()):
                        raise ValueError(f"验证指标包含非法数值：{source.name}")
                    if not all(math.isfinite(values[name]) for name in ("ssim", "lpips")):
                        raise ValueError(f"验证 SSIM 或 LPIPS 包含非有限数值：{source.name}")
                    results.append(values)
        finally:
            # 直接还原各模块，避免 model.train() 覆盖 Stage 2 冻结分支的 eval 状态。
            for module, was_training in training_states:
                module.training = was_training
            restore_rng_state(rng_state)
        return {
            **{name: sum(result[name] for result in results) / len(results)
               for name in ("psnr", "ssim", "lpips")},
            "num_images": len(results),
        }


class MultiDatasetValidator:
    """按配置顺序评价多个数据集，并共享一份官方冻结 LPIPS 网络。

    配置形如 ``datasets: {LOLv1: {low_dir: ..., high_dir: ...}, ...}``。
    所有数据集使用顶层 ``lpips_net``；每个数据集仍分别按图像平均指标。
    不合并不同数据集的分数，不缩放原图，不保存最佳训练状态或权重。
    Trainer 负责整个轮末验证的训练状态与随机状态保护。
    """

    def __init__(self, config: dict, device: torch.device | str):
        self.device = torch.device(device)
        self.lpips_net = config["lpips_net"]
        if self.lpips_net not in ("alex", "vgg", "squeeze"):
            raise ValueError("validation.lpips_net 必须是 alex、vgg 或 squeeze")
        datasets = config.get("datasets")
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValueError("validation.datasets 必须是非空的数据集名称到目录配置的映射")

        dataset_configs = {}
        used_names = set()
        for name, dataset in datasets.items():
            # 名称直接用于最佳权重目录，必须限制为一个普通路径组件。
            if (not isinstance(name, str) or not name or name.strip() != name
                    or name in (".", "..") or "/" in name or "\\" in name
                    or "\x00" in name):
                raise ValueError("validation.datasets 的名称必须是非空的单个路径组件")
            if name.casefold() in used_names:
                raise ValueError("validation.datasets 不允许大小写重复的数据集名称")
            used_names.add(name.casefold())
            if not isinstance(dataset, Mapping) or not all(key in dataset for key in ("low_dir", "high_dir")):
                raise ValueError(f"验证数据集 {name} 必须配置 low_dir 和 high_dir")
            dataset_config = {**dataset, "lpips_net": self.lpips_net}
            low_dir = Path(normalize_path(dataset_config["low_dir"]))
            high_dir = Path(normalize_path(dataset_config["high_dir"]))
            # 先完整检查所有数据集，避免后面的坏配对触发无意义模型构造。
            _paired_image_paths(low_dir, high_dir)
            dataset_configs[name] = dataset_config

        self.lpips_metric = _create_lpips_model(self.lpips_net, self.device)
        self.validators = {
            name: EpochValidator(dataset, self.device, lpips_model=self.lpips_metric)
            for name, dataset in dataset_configs.items()
        }

    def signature(self) -> dict:
        """返回各数据集原有指纹，保留配置顺序并与内部状态隔离。"""
        return {name: validator.signature() for name, validator in self.validators.items()}

    def validate(self, model: torch.nn.Module, mode: str) -> dict:
        """逐个推理，返回数据集名称到各自四项评价结果的映射。"""
        return {name: validator.validate(model, mode) for name, validator in self.validators.items()}
