"""验证共享配置、推理阶段选择以及排除图像读写的 CPU/CUDA 前向计时。"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

from inference import test as inference_cli
from models import SIMI
from utils.config import DEFAULT_CONFIG, PROJECT_ROOT, load_yaml, normalize_path, normalize_training_paths
from utils.image_utils import load_image, save_image
from utils import timing


class InferenceRuntimeTests(unittest.TestCase):
    """只使用临时图像和小型前向，不运行正式训练。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def write_config(self, directory, values):
        """将测试需要的 YAML 写入独立临时目录。"""
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return path

    def test_parser_defaults_defer_to_unified_yaml(self):
        """解析命令行时不抢占 YAML 值，必需路径在合并之后检查。"""
        args = inference_cli.build_parser().parse_args([])
        self.assertEqual(args.config, DEFAULT_CONFIG)
        for key in ("checkpoint", "input", "output", "gt", "device", "stage", "benchmark", "warmup"):
            self.assertIsNone(getattr(args, key))
        with self.assertRaisesRegex(ValueError, "inference.checkpoint"):
            inference_cli.resolve_inference_args(args)

    def test_yaml_controls_device_paths_stage_timing_and_experiments(self):
        """YAML 相对路径以项目根目录解析，并展开用户目录。"""
        config = {
            "runtime": {"device": "cpu"}, "bit_mode": "gray",
            "enhancement_guidance": "snr", "enhancement_backbone": "legacy_cnn",
            "inference": {
                "checkpoint": "weights/sample.pth", "input": "images/low",
                "output": "~/simi-runtime-test-results", "gt": "images/high",
                "stage": "enhancement", "benchmark": False, "warmup": 7,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, config)
            args = inference_cli.build_parser().parse_args(["--config", str(path)])
            resolved = inference_cli.resolve_inference_args(args)
        self.assertEqual(resolved.device, "cpu")
        self.assertEqual(resolved.checkpoint, (PROJECT_ROOT / "weights/sample.pth").resolve())
        self.assertEqual(resolved.input, (PROJECT_ROOT / "images/low").resolve())
        self.assertEqual(resolved.gt, (PROJECT_ROOT / "images/high").resolve())
        self.assertEqual(resolved.output, (Path.home() / "simi-runtime-test-results").resolve())
        self.assertEqual((resolved.stage, resolved.benchmark, resolved.warmup), ("enhancement", False, 7))
        self.assertEqual(inference_cli._experiment_options(resolved), {
            "bit_mode": "gray", "enhancement_guidance": "snr", "enhancement_backbone": "legacy_cnn",
        })

    def test_cli_overrides_yaml_and_resolves_paths_against_cwd(self):
        """全部命令行覆盖项优先，命令行相对路径使用当前工作目录。"""
        config = {
            "runtime": {"device": "cuda:7"}, "bit_mode": "gray",
            "enhancement_guidance": "snr", "enhancement_backbone": "legacy_cnn",
            "inference": {
                "checkpoint": "yaml.pth", "input": "yaml-input", "output": "yaml-output",
                "gt": "yaml-gt", "stage": "denoising", "benchmark": False, "warmup": 9,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, config)
            current_directory = Path(directory) / "caller"
            args = inference_cli.build_parser().parse_args([
                "--config", str(path), "--checkpoint", "cli/model.pth", "--input", "cli/input",
                "--output", "~/simi-cli-test-results", "--gt", "cli/gt", "--device", "cpu",
                "--stage", "enhancement", "--benchmark", "--warmup", "0",
                "--bit-mode", "binary", "--enhancement-guidance", "none", "--enhancement-backbone", "dea",
            ])
            with patch.object(Path, "cwd", return_value=current_directory):
                resolved = inference_cli.resolve_inference_args(args)
        self.assertEqual(resolved.checkpoint, current_directory / "cli/model.pth")
        self.assertEqual(resolved.input, current_directory / "cli/input")
        self.assertEqual(resolved.gt, current_directory / "cli/gt")
        self.assertEqual(resolved.output, (Path.home() / "simi-cli-test-results").resolve())
        self.assertEqual((resolved.device, resolved.stage, resolved.benchmark, resolved.warmup), ("cpu", "enhancement", True, 0))
        self.assertEqual(inference_cli._experiment_options(resolved), {
            "bit_mode": "binary", "enhancement_guidance": "none", "enhancement_backbone": "dea",
        })

    def test_shared_path_normalization_preserves_empty_and_nonexistent_paths(self):
        """训练与推理复用路径规则，输出路径可以在运行时才创建。"""
        config = {"data": {"root": "images"}, "train": {"snapshots_folder": "~/simi-empty-output", "pretrain_dir": ""}}
        self.assertIs(normalize_training_paths(config), config)
        self.assertEqual(config["data"]["root"], str((PROJECT_ROOT / "images").resolve()))
        self.assertEqual(config["train"]["snapshots_folder"], str((Path.home() / "simi-empty-output").resolve()))
        self.assertEqual(config["train"]["pretrain_dir"], "")
        self.assertIsNone(normalize_path(None))
        self.assertEqual(normalize_path(""), "")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("- one\n- two\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YAML 映射"):
                load_yaml(path)

    def test_invalid_configuration_fails_before_model_construction(self):
        """错误阶段、设备、计时开关和预热次数不能静默生效。"""
        for key, invalid in (("stage", "joint"), ("benchmark", "false"), ("warmup", -1), ("warmup", True), ("checkpoint", 123)):
            with self.subTest(key=key, invalid=invalid), tempfile.TemporaryDirectory() as directory:
                values = {"checkpoint": "model.pth", "input": "image.png", "output": "result.png", key: invalid}
                config = self.write_config(directory, {"runtime": {"device": "cpu"}, "inference": values})
                args = inference_cli.build_parser().parse_args(["--config", str(config)])
                with patch.object(inference_cli, "SIMI") as constructor, self.assertRaises(ValueError):
                    inference_cli.run_inference(args)
                constructor.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            config = self.write_config(directory, {"runtime": {"device": 0}})
            args = inference_cli.build_parser().parse_args(["--config", str(config)])
            with self.assertRaisesRegex(ValueError, "runtime.device"):
                inference_cli.resolve_inference_args(args)

    def exercise_inference(self, stage, *, device="cpu", benchmark=True, warmup=2, images=2):
        """对实际模型记录分支调用，并和未经计时包装的输出比较。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            inputs, outputs, targets = directory / "input", directory / "output", directory / "gt"
            inputs.mkdir()
            targets.mkdir()
            for index in range(images):
                pixels = np.random.default_rng(91 + index).integers(0, 256, (16, 20, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(inputs / f"{index}.png")
                Image.fromarray(pixels).save(targets / f"{index}.png")
            model = SIMI().to(device).eval()
            checkpoint = directory / "model.pth"
            torch.save({"net": model.state_dict(), "model_config": model.checkpoint_config()}, checkpoint)
            expected = {}
            with torch.inference_mode():
                for source in sorted(inputs.iterdir()):
                    rgb = load_image(source).unsqueeze(0).to(device)
                    planes = model.bitplane_extractor(rgb)
                    expected[source.name] = model.light_net(rgb, planes)[0] if stage == "enhancement" else model(rgb, planes)[3]
            arguments = [
                "--checkpoint", str(checkpoint), "--input", str(inputs), "--output", str(outputs),
                "--gt", str(targets), "--device", device, "--stage", stage, "--warmup", str(warmup),
                "--benchmark" if benchmark else "--no-benchmark",
            ]
            args = inference_cli.build_parser().parse_args(arguments)
            light_forward = model.light_net.forward
            states = []

            def record_light(*forward_args, **kwargs):
                """记录评估模式与无梯度推理上下文，不改变计算结果。"""
                states.append((model.training, torch.is_inference_mode_enabled()))
                return light_forward(*forward_args, **kwargs)

            printed = io.StringIO()
            with patch.object(inference_cli, "SIMI", return_value=model) as constructor, \
                    patch.object(inference_cli.torch, "load", wraps=torch.load) as loader, \
                    patch.object(inference_cli, "load_yaml", wraps=load_yaml) as config_loader, \
                    patch.object(model.light_net, "forward", side_effect=record_light) as light, \
                    patch.object(model.denoise, "forward", wraps=model.denoise.forward) as denoiser, \
                    patch.object(model.bitplane_extractor, "forward", wraps=model.bitplane_extractor.forward) as extractor, \
                    patch.object(inference_cli, "image_metrics", wraps=inference_cli.image_metrics) as metrics, \
                    redirect_stdout(printed):
                result = inference_cli.run_inference(args)
            constructor.assert_called_once()
            loader.assert_called_once()
            config_loader.assert_called_once()
            self.assertEqual(extractor.call_count, images)
            self.assertEqual(metrics.call_count, images)
            forwards = images * (warmup + 1 if benchmark else 1)
            self.assertEqual(light.call_count, forwards)
            self.assertEqual(denoiser.call_count, forwards if stage == "denoising" else 0)
            self.assertTrue(all(state == (False, True) for state in states))
            self.assertIn("psnr", result)
            self.assertIn("ssim", result)
            self.assertIn(f"Mean over {images} images", printed.getvalue())
            if benchmark:
                self.assertEqual(len(result["timings"]), images)
                self.assertTrue(all(value > 0 for value in result["timings"]))
                self.assertEqual(result["timing"], sum(result["timings"]) / images)
                self.assertIn(f"Mean forward over {images} images ({stage})", printed.getvalue())
            else:
                self.assertNotIn("timing", result)
                self.assertNotIn("timings", result)
            for name, output in expected.items():
                expected_path = directory / f"expected_{name}"
                save_image(output, expected_path)
                with Image.open(outputs / name) as actual_image, Image.open(expected_path) as expected_image:
                    self.assertTrue(np.array_equal(np.array(actual_image), np.array(expected_image)))

    def test_cpu_timing_preserves_outputs_and_selected_stage_call_counts(self):
        """CPU 两阶段均只计前向，预热不增加图像数或指标次数。"""
        for stage in ("enhancement", "denoising"):
            with self.subTest(stage=stage):
                self.exercise_inference(stage)

    def test_old_cli_without_explicit_config_and_no_benchmark_remains_available(self):
        """旧完整命令行继续可用；关闭计时时不执行配置中的预热。"""
        self.exercise_inference("enhancement", benchmark=False, warmup=5, images=1)

    @unittest.skipUnless(torch.cuda.is_available(), "需要可用 CUDA 设备")
    def test_cuda_timing_is_positive_and_preserves_both_stage_outputs(self):
        """实际 CUDA 推理完成同步计时，并保持两个阶段的输出一致。"""
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            for stage in ("enhancement", "denoising"):
                with self.subTest(stage=stage):
                    self.exercise_inference(stage, device="cuda:0", warmup=1, images=1)

    def test_timing_boundary_contains_only_forward_and_cuda_synchronizations(self):
        """验证计时顺序，CPU 不调用 CUDA 同步，CUDA 两侧均同步指定设备。"""
        for device in ("cpu", "cuda:2"):
            with self.subTest(device=device):
                events = []
                clocks = iter((10.0, 10.25))

                def clock_value():
                    """在时钟读取位置记录事件。"""
                    events.append("clock")
                    return next(clocks)

                def forward():
                    """只模拟所选网络前向，不包含图像读写。"""
                    events.append("forward")
                    return "output"

                def synchronize(selected_device):
                    """确认同步的是显式选择的 CUDA 设备。"""
                    self.assertEqual(selected_device, torch.device(device))
                    events.append("sync")

                with patch.object(timing.time, "perf_counter", side_effect=clock_value), \
                        patch.object(timing.torch.cuda, "synchronize", side_effect=synchronize):
                    output, duration = timing.measure_forward(forward, device)
                self.assertEqual((output, duration), ("output", 0.25))
                self.assertEqual(events, ["clock", "forward", "clock"] if device == "cpu" else ["sync", "clock", "forward", "sync", "clock"])
        for invalid in (-1, True, 1.5):
            with self.subTest(warmup=invalid), self.assertRaises(ValueError):
                timing.benchmark_forward(lambda: None, "cpu", warmup=invalid)


if __name__ == "__main__":
    unittest.main()
