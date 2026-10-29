"""使用 SIMI 检查点处理单张图片或整个目录，保持图片原始尺寸。"""

import argparse
from pathlib import Path
import sys


# 同时支持 `python inference/test.py` 和 `python -m inference.test` 两种执行方式。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from datasets import BIT_MODES
from models import ENHANCEMENT_BACKBONES, ENHANCEMENT_GUIDANCE_MODES, SIMI
from utils.config import DEFAULT_CONFIG, PROJECT_ROOT, load_yaml, normalize_path
from utils.image_utils import image_to_tensor, load_image, read_rgb_array, save_image
from utils.metrics import average_metrics, image_metrics
from utils.timing import benchmark_forward


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="统一 YAML 配置文件")
    parser.add_argument("--checkpoint", type=Path, help="覆盖 inference.checkpoint")
    parser.add_argument("--input", type=Path, help="输入图像或目录，覆盖 inference.input")
    parser.add_argument("--output", type=Path, help="输出图像或目录，覆盖 inference.output")
    parser.add_argument("--device", help="覆盖 runtime.device，例如 cuda:0 或 cpu")
    parser.add_argument("--stage", choices=("enhancement", "denoising"), help="覆盖 inference.stage")
    parser.add_argument("--gt", type=Path, help="参考图像或同名参考图像目录")
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--benchmark", dest="benchmark", action="store_true", help="启用模型前向计时")
    timing.add_argument("--no-benchmark", dest="benchmark", action="store_false", help="关闭模型前向计时")
    parser.set_defaults(benchmark=None)
    parser.add_argument("--warmup", type=int, help="每张图在所选阶段执行的预热次数")
    parser.add_argument("--bit-mode", choices=BIT_MODES, help="覆盖 YAML bit_mode")
    parser.add_argument("--enhancement-guidance", choices=ENHANCEMENT_GUIDANCE_MODES,
                        help="覆盖 YAML enhancement_guidance")
    parser.add_argument("--enhancement-backbone", choices=ENHANCEMENT_BACKBONES,
                        help="覆盖 YAML 主干；旧检查点需选择 legacy_cnn")
    return parser


def _experiment_options(args: argparse.Namespace) -> dict:
    """读取研究和结构选项，命令行优先，默认使用修正后的 DEA 网络。"""
    config = getattr(args, "_config", None)
    if config is None:
        config_path = getattr(args, "config", None)
        config = load_yaml(config_path) if config_path is not None else {}
    options = {}
    for key, default in (("bit_mode", "binary"), ("enhancement_guidance", "none"),
                         ("enhancement_backbone", "dea")):
        override = getattr(args, key, None)
        options[key] = config.get(key, default) if override is None else override
    return options


def resolve_inference_args(args: argparse.Namespace) -> argparse.Namespace:
    """合并 YAML 与命令行，优先使用显式命令行值，并统一路径语义。

    YAML 相对路径基于项目根目录，命令行相对路径基于调用者工作目录。
    旧的完整命令行调用方式不需要显式传入 --config，默认读取项目配置。
    """
    if getattr(args, "_resolved", False):
        return args
    resolved = argparse.Namespace(**vars(args))
    config_path = getattr(args, "config", None)
    config = load_yaml(config_path) if config_path is not None else {}
    runtime = config.get("runtime", {})
    inference = config.get("inference", {})
    if not isinstance(runtime, dict) or not isinstance(inference, dict):
        raise ValueError("runtime 和 inference 配置必须是 YAML 映射")
    resolved._config = config
    for key, value in _experiment_options(resolved).items():
        setattr(resolved, key, value)
    defaults = {
        "checkpoint": "", "input": "", "output": "results", "gt": "",
        "stage": "denoising", "benchmark": True, "warmup": 3,
    }
    for key, default in defaults.items():
        override = getattr(args, key, None)
        value = inference.get(key, default) if override is None else override
        if key in ("checkpoint", "input", "output", "gt"):
            if value is not None and not isinstance(value, (str, Path)):
                raise ValueError(f"inference.{key} 必须是路径字符串")
            normalized = normalize_path(value, base_dir=PROJECT_ROOT if override is None else Path.cwd())
            value = Path(normalized) if normalized else None
        setattr(resolved, key, value)
    resolved.device = getattr(args, "device", None)
    if resolved.device is None:
        resolved.device = runtime.get("device", "cuda:0")
    if not isinstance(resolved.device, str) or not resolved.device.strip():
        raise ValueError("runtime.device 必须是非空设备字符串")
    if resolved.stage not in ("enhancement", "denoising"):
        raise ValueError("inference.stage 必须是 enhancement 或 denoising")
    if not isinstance(resolved.benchmark, bool):
        raise ValueError("inference.benchmark 必须是布尔值")
    if isinstance(resolved.warmup, bool) or not isinstance(resolved.warmup, int) or resolved.warmup < 0:
        raise ValueError("inference.warmup 必须是非负整数")
    for key in ("checkpoint", "input", "output"):
        if getattr(resolved, key) is None:
            raise ValueError(f"请通过 inference.{key} 或 --{key} 提供必需路径")
    resolved._resolved = True
    return resolved


def _input_images(path: Path) -> list:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    images = sorted(
        candidate for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in {path}")
    return images


def run_inference(args: argparse.Namespace) -> dict:
    args = resolve_inference_args(args)
    images = _input_images(args.input)
    output_is_file = args.input.is_file() and args.output.suffix.lower() in IMAGE_EXTENSIONS
    if args.gt is not None:
        if args.gt.is_file() and len(images) != 1:
            raise ValueError("A ground-truth file can only be used with one input image")
        if not args.gt.exists():
            raise FileNotFoundError(f"Ground truth does not exist: {args.gt}")
        for source in images:
            target = args.gt / source.name if args.gt.is_dir() else args.gt
            if not target.is_file():
                raise FileNotFoundError(f"Missing matching ground truth: {target}")

    device = torch.device(args.device)
    model = SIMI(**_experiment_options(args)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.validate_checkpoint_config(checkpoint)
    state = checkpoint["net"] if "net" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    results = []
    timings = []
    with torch.inference_mode():
        for source in images:
            pixels = read_rgb_array(source)
            rgb = image_to_tensor(pixels).unsqueeze(0).to(device)
            integer_rgb = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
            # 推理与训练使用同一模型所选择的提取器，不绕过实验配置。
            planes = model.bitplane_extractor(integer_rgb)
            if args.stage == "enhancement":
                # 只运行增强分支，避免生成不会使用的去噪结果。
                forward = lambda: model.light_net(rgb, planes)[0]
            else:
                # 去噪需要增强图和末级特征，保留原完整前向流程。
                forward = lambda: model(rgb, planes)[3]
            if args.benchmark:
                output, elapsed = benchmark_forward(forward, device, warmup=args.warmup)
                timings.append(elapsed)
                timing_text = f" forward={elapsed:.6f}s"
            else:
                output = forward()
                timing_text = ""
            destination = args.output if output_is_file else args.output / source.name
            save_image(output, destination)
            if args.gt is not None:
                target_path = args.gt / source.name if args.gt.is_dir() else args.gt
                target = load_image(target_path).unsqueeze(0).to(device)
                metrics = image_metrics(output, target)
                results.append(metrics)
                print(
                    f"{source.name}: PSNR={metrics['psnr'].item():.4f} "
                    f"SSIM={metrics['ssim'].item():.4f}{timing_text}"
                )
            else:
                print(f"Saved {destination}{timing_text}")
    summary = {}
    if results:
        mean = average_metrics(results)
        print(f"Mean over {len(results)} images: PSNR={mean['psnr']:.4f} SSIM={mean['ssim']:.4f}")
        summary.update(mean)
    if timings:
        summary["timing"] = sum(timings) / len(timings)
        summary["timings"] = timings
        print(f"Mean forward over {len(timings)} images ({args.stage}): {summary['timing']:.6f}s")
    return summary


def main(argv=None) -> dict:
    return run_inference(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
