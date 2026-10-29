"""通过 YAML 选择 SIMI 增强、去噪或保持原始更新顺序的联合训练。"""

import argparse
from datetime import datetime
from pathlib import Path
import time

from datasets import BIT_MODES
from models import ENHANCEMENT_BACKBONES, ENHANCEMENT_GUIDANCE_MODES
from trainers.trainer import Trainer
from utils.config import PROJECT_ROOT, DEFAULT_CONFIG, load_yaml, normalize_path, normalize_training_paths
from utils.logging import tee_training_log, validate_output_paths


def load_config(path):
    """以项目目录为基准解析数据和输出路径，不依赖当前工作目录。"""
    config = load_yaml(path)
    Trainer.validate_config(config)
    return normalize_training_paths(config)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="Override runtime.device, e.g. cuda:0 or cpu")
    parser.add_argument("--mode", choices=Trainer.MODES, help="Override train.mode")
    parser.add_argument("--bit-mode", choices=BIT_MODES, help="Override bit_mode")
    parser.add_argument("--enhancement-guidance", choices=ENHANCEMENT_GUIDANCE_MODES,
                        help="Override enhancement_guidance")
    parser.add_argument("--enhancement-backbone", choices=ENHANCEMENT_BACKBONES,
                        help="Override backbone; legacy_cnn reproduces old checkpoints")
    parser.add_argument("--data-root", type=Path, help="Override the training image root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir", type=Path, help="覆盖 output.root；日志和权重统一写入此目录")
    parser.add_argument("--snapshots-folder", type=Path)
    parser.add_argument("--pretrain", type=Path, help="Load checkpoint according to train.resume_optimizers")
    parser.add_argument("--resume", type=Path, help="Fully resume model, progress, schedulers and RNG")
    parser.add_argument("--resume-mode", choices=("full", "weights", "legacy"))
    parser.add_argument("--seed", type=int, help="Override runtime.seed")
    parser.add_argument("--max-steps", type=int, help="Save and stop at this absolute global step")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--amp", dest="amp", action="store_true", help="启用 CUDA 自动混合精度")
    precision.add_argument("--no-amp", dest="amp", action="store_false", help="关闭 AMP，使用 FP32 训练")
    parser.set_defaults(amp=None)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), help="覆盖 amp.dtype")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    # 旧单数据集 YAML 仍可读取；新入口为日志提供统一的输出根目录。
    config.setdefault("output", {"root": normalize_path("outputs")})
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.seed is not None:
        config["runtime"]["seed"] = args.seed
    if args.amp is not None:
        config["amp"]["enabled"] = args.amp
    if args.amp_dtype is not None:
        config["amp"]["dtype"] = args.amp_dtype
    if args.mode is not None:
        config["train"]["mode"] = args.mode
    if args.bit_mode is not None:
        config["bit_mode"] = args.bit_mode
    if args.enhancement_guidance is not None:
        config["enhancement_guidance"] = args.enhancement_guidance
    if args.enhancement_backbone is not None:
        config["enhancement_backbone"] = args.enhancement_backbone
    if args.data_root is not None:
        config["data"]["root"] = normalize_path(args.data_root, base_dir=Path.cwd())
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    for name, key in (("epochs", "epochs"), ("batch_size", "batch_size")):
        value = getattr(args, name)
        if value is not None:
            config["train"][key] = value
    if args.output_dir is not None:
        config["output"]["root"] = normalize_path(args.output_dir, base_dir=Path.cwd())
        config["train"]["snapshots_folder"] = str(Path(config["output"]["root"]) / "weights")
    if args.snapshots_folder is not None:
        config["train"]["snapshots_folder"] = normalize_path(args.snapshots_folder, base_dir=Path.cwd())
    if args.pretrain is not None:
        config["train"]["load_pretrain"] = True
        config["train"]["pretrain_dir"] = normalize_path(args.pretrain, base_dir=Path.cwd())
    if args.resume is not None:
        config["train"].update(load_pretrain=True, resume_mode="full", resume_optimizers=True,
                               pretrain_dir=normalize_path(args.resume, base_dir=Path.cwd()))
    if args.resume_mode is not None:
        config["train"]["resume_mode"] = args.resume_mode
        config["train"]["resume_optimizers"] = args.resume_mode != "weights"
    # 所有覆盖项先经过检查，非法配置不会创建日志目录或启动模型。
    Trainer.validate_config(config)
    validate_output_paths(config["output"]["root"], config["train"]["snapshots_folder"])
    start = time.perf_counter()
    with tee_training_log(config["output"]["root"]) as log_path:
        print("=" * 80, flush=True)
        print(f"训练运行开始：{datetime.now().astimezone().isoformat(timespec='seconds')}", flush=True)
        print(f"配置文件：{args.config.expanduser().resolve()}", flush=True)
        print(f"日志文件：{log_path}", flush=True)
        print(f"权重目录：{config['train']['snapshots_folder']}", flush=True)
        trainer = Trainer(config)
        print(
            f"训练精度：{'AMP ' + trainer.amp_dtype_name if trainer.amp_enabled else 'FP32'}；"
            f"梯度缩放：{'启用' if trainer.scaler.is_enabled() else '关闭'}", flush=True,
        )
        trainer.fit(max_steps=args.max_steps)
        if trainer.device.type == "cuda":
            import torch
            torch.cuda.synchronize(trainer.device)
        print(f"Training finished in {time.perf_counter() - start:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
