"""验证实验模式工厂、显式模块注入以及训练和推理配置入口。"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import inspect
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from datasets import (
    BIT_MODES,
    BinaryBitPlaneExtractor,
    BitPlaneExtractor,
    build_bitplane_extractor,
    validate_bit_mode,
)
from inference import test as inference_cli
from models import (
    ENHANCEMENT_GUIDANCE_MODES,
    CurveEnhancementNetwork,
    EnhancementNetwork,
    SIMI,
    build_enhancement_network,
    validate_enhancement_guidance,
)
import train as training_cli
from trainers.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]


class RecordingExtractor(BinaryBitPlaneExtractor):
    """记录提取器调用，仍沿用现有二进制算法，不实现新的实验算法。"""

    def __init__(self):
        super().__init__()
        self.inputs = []
        self.outputs = []

    def forward(self, image):
        self.inputs.append(image)
        output = super().forward(image)
        self.outputs.append(output)
        return output


class RecordingEnhancement(EnhancementNetwork):
    """记录位平面和引导信息，仅运行原增强流程，不实现引导算法。"""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, rgb, bitplanes, extra_guidance=None):
        self.calls.append((rgb, bitplanes, extra_guidance))
        return super().forward(rgb, bitplanes, extra_guidance=None)


class ExperimentInterfaceTests(unittest.TestCase):
    """用无参数记录模块验证接口接线与默认数学行为。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self):
        """读取完整 YAML，并将测试运行设备限定为 CPU。"""
        config = training_cli.load_config(ROOT / "configs/default.yaml")
        config["validation"]["enabled"] = False
        config["runtime"]["device"] = "cpu"
        config["train"]["load_pretrain"] = False
        config["train"]["pretrain_dir"] = ""
        config["data"]["num_workers"] = 0
        config["data"]["pin_memory"] = False
        return config

    def image(self):
        """生成非正方形小批次，覆盖空间和批次维的保留行为。"""
        return torch.rand((2, 3, 13, 17), generator=torch.Generator().manual_seed(661))

    def assert_state_equal(self, actual, expected):
        """同时比较状态顺序和逐元素数值。"""
        self.assertEqual(list(actual), list(expected))
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_declared_modes_and_factories_never_fall_back_silently(self):
        """未实现的合法模式明确报错，非法名称也不能回退到默认算法。"""
        self.assertEqual(BIT_MODES, ("binary", "gray", "learnable", "reliable"))
        self.assertEqual(ENHANCEMENT_GUIDANCE_MODES, ("none", "snr", "reliability"))
        self.assertIsInstance(build_bitplane_extractor(), BinaryBitPlaneExtractor)
        self.assertIsInstance(build_enhancement_network(), EnhancementNetwork)
        for mode in BIT_MODES:
            validate_bit_mode(mode)
            if mode != "binary":
                with self.subTest(bit_mode=mode), self.assertRaises(NotImplementedError):
                    build_bitplane_extractor(mode)
        for mode in ENHANCEMENT_GUIDANCE_MODES:
            validate_enhancement_guidance(mode)
            if mode != "none":
                with self.subTest(enhancement_guidance=mode), self.assertRaises(NotImplementedError):
                    build_enhancement_network(mode)
        for invalid in ("unknown", "", "BINARY", "GRAY"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_bit_mode(invalid)
                with self.assertRaises(ValueError):
                    build_bitplane_extractor(invalid)
                with self.assertRaises(ValueError):
                    validate_enhancement_guidance(invalid)
                with self.assertRaises(ValueError):
                    build_enhancement_network(invalid)

    def test_factory_injection_preserves_identity_and_checks_module_contract(self):
        """合法显式注入返回原对象，增强网络必须兼容原四项输出约定。"""
        extractor = RecordingExtractor()
        network = RecordingEnhancement()
        for mode in BIT_MODES:
            self.assertIs(build_bitplane_extractor(mode, extractor=extractor), extractor)
        for mode in ENHANCEMENT_GUIDANCE_MODES:
            self.assertIs(build_enhancement_network(mode, network=network), network)
        for invalid in (object(), nn.Identity(), CurveEnhancementNetwork()):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises(TypeError):
                    build_bitplane_extractor("binary", extractor=invalid)
                with self.assertRaises(TypeError):
                    build_enhancement_network("none", network=invalid)
        with self.assertRaises(ValueError):
            build_bitplane_extractor("unknown", extractor=extractor)
        with self.assertRaises(ValueError):
            build_enhancement_network("unknown", network=network)

    def test_implicit_and_explicit_defaults_preserve_parameters_forward_and_gradients(self):
        """缺省模式和显式默认模式保持初始化、参数顺序、输出及全部梯度一致。"""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(211)
            implicit = SIMI()
            torch.manual_seed(211)
            explicit = SIMI(bit_mode="binary", enhancement_guidance="none", enhancement_backbone="dea")
        self.assert_state_equal(implicit.state_dict(), explicit.state_dict())
        with self.assertRaisesRegex(RuntimeError, "enhancement_backbone=legacy_cnn"):
            implicit.legacy_state_dict()
        self.assertEqual(sum(parameter.numel() for parameter in implicit.parameters()), 488044)
        self.assertEqual([name for name, _ in implicit.named_parameters()], [name for name, _ in explicit.named_parameters()])
        left_image = self.image().requires_grad_()
        right_image = left_image.detach().clone().requires_grad_()
        left_outputs = implicit(left_image)
        right_outputs = explicit(right_image)
        self.assertEqual(len(left_outputs), 4)
        for left, right in zip(left_outputs, right_outputs):
            self.assertTrue(torch.equal(left, right))
        sum(output.sum() for output in left_outputs).backward()
        sum(output.sum() for output in right_outputs).backward()
        self.assertTrue(torch.equal(left_image.grad, right_image.grad))
        for left, right in zip(implicit.parameters(), explicit.parameters()):
            self.assertEqual(left.grad is None, right.grad is None)
            if left.grad is not None:
                self.assertTrue(torch.equal(left.grad, right.grad))

    def test_model_owns_one_extractor_and_forwards_planes_and_guidance(self):
        """模型统一持有提取器，默认提取、显式位平面和引导信息均正确接线。"""
        extractor, network = RecordingExtractor(), RecordingEnhancement()
        model = SIMI("reliable", "reliability", bitplane_extractor=extractor, enhancement_network=network)
        self.assertIs(model.bitplane_extractor, extractor)
        self.assertIs(model.light_net, network)
        self.assertEqual(list(model._modules), ["light_net", "denoise", "bitplane_extractor"])
        self.assertEqual(sum(module is extractor for _, module in model.named_modules(remove_duplicate=False)), 1)
        self.assertEqual(list(extractor.parameters()), [])
        self.assertEqual(list(extractor.state_dict()), [])
        self.assertFalse(any(key.startswith("bitplane_extractor.") for key in model.state_dict()))
        image, guidance = self.image(), torch.ones(2, 1, 13, 17)
        with torch.no_grad():
            outputs = model(image, extra_guidance=guidance)
            self.assertEqual(len(outputs), 4)
            self.assertEqual(len(extractor.inputs), 1)
            self.assertIs(extractor.inputs[0], image)
            self.assertIs(network.calls[-1][1], extractor.outputs[0])
            self.assertIs(network.calls[-1][2], guidance)
            planes = extractor.outputs[0]
            final = model.enhance(image, planes, extra_guidance=guidance)
            self.assertTrue(torch.equal(final, outputs[3]))
            self.assertEqual(len(extractor.inputs), 1)
            features = model.extract_features(image)
            self.assertEqual(len(extractor.inputs), 2)
            self.assertIs(features["raw_bitplanes"], extractor.outputs[-1])

    def test_unimplemented_model_modes_require_explicit_injection(self):
        """模型本身也不能为未实现的实验模式静默创建默认模块。"""
        for mode in BIT_MODES[1:]:
            with self.subTest(bit_mode=mode), self.assertRaises(NotImplementedError):
                SIMI(bit_mode=mode)
        for mode in ENHANCEMENT_GUIDANCE_MODES[1:]:
            with self.subTest(enhancement_guidance=mode), self.assertRaises(NotImplementedError):
                SIMI(enhancement_guidance=mode)

    def test_all_training_modes_forward_guidance_to_injected_network(self):
        """三种训练模式共用模型提取器，并透传引导信息而不新增引导算法。"""
        image, guidance = self.image(), torch.ones(2, 1, 13, 17)
        for mode in ("enhancement", "denoising", "joint"):
            with self.subTest(mode=mode):
                config = self.config()
                config.update(bit_mode="gray", enhancement_guidance="snr")
                config["train"]["mode"] = mode
                extractor, network = RecordingExtractor(), RecordingEnhancement()
                trainer = Trainer(config, bitplane_extractor=extractor, enhancement_network=network)
                self.assertIs(trainer.bitplane_extractor, extractor)
                self.assertIs(trainer.model.bitplane_extractor, extractor)
                self.assertEqual(sum(module is extractor for _, module in trainer.model.named_modules(remove_duplicate=False)), 1)
                trainer.train_step(image, extra_guidance=guidance)
                self.assertEqual(len(extractor.inputs), 1)
                self.assertEqual(len(network.calls), 1)
                self.assertIs(network.calls[0][1], extractor.outputs[0])
                self.assertIs(network.calls[0][2], guidance)
                explicit = extractor.outputs[0]
                trainer.train_step(image, explicit, extra_guidance=guidance)
                self.assertEqual(len(extractor.inputs), 1)
                self.assertEqual(len(network.calls), 2)
                self.assertIs(network.calls[-1][1], explicit)
                self.assertIs(network.calls[-1][2], guidance)

    def test_trainer_rejects_parameterized_extractor(self):
        """尚未实现提取器优化策略时，拒绝存在可训练参数的注入模块。"""
        class ParameterizedRecordingExtractor(RecordingExtractor):
            """只添加测试参数，不实现新的位平面算法。"""

            def __init__(self):
                super().__init__()
                self.test_parameter = nn.Parameter(torch.zeros(()))

        for mode in ("enhancement", "denoising", "joint"):
            with self.subTest(mode=mode):
                config = self.config()
                config["train"]["mode"] = mode
                config["bit_mode"] = "learnable"
                with self.assertRaises(NotImplementedError):
                    Trainer(config, bitplane_extractor=ParameterizedRecordingExtractor())

    def test_yaml_experiment_choices_and_required_top_level_fields(self):
        """完整配置允许声明未来模式，但缺少或误写顶层模式必须报错。"""
        config = self.config()
        self.assertEqual(config["bit_mode"], "binary")
        self.assertEqual(config["enhancement_guidance"], "none")
        for bit_mode in BIT_MODES:
            for guidance_mode in ENHANCEMENT_GUIDANCE_MODES:
                candidate = deepcopy(config)
                candidate.update(bit_mode=bit_mode, enhancement_guidance=guidance_mode)
                Trainer.validate_config(candidate)
        for key in ("bit_mode", "enhancement_guidance"):
            with self.subTest(key=key):
                missing = deepcopy(config)
                del missing[key]
                with self.assertRaises(ValueError):
                    Trainer.validate_config(missing)
                invalid = deepcopy(config)
                invalid[key] = "unknown"
                with self.assertRaises(ValueError):
                    Trainer.validate_config(invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            candidate = deepcopy(config)
            candidate.update(bit_mode="gray", enhancement_guidance="snr")
            path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            loaded = training_cli.load_config(path)
            self.assertEqual(loaded["bit_mode"], "gray")
            self.assertEqual(loaded["enhancement_guidance"], "snr")
            with self.assertRaises(NotImplementedError):
                Trainer(loaded)

    def test_training_and_inference_cli_validate_mode_choices(self):
        """命令行接受已声明的枚举并拒绝非法名称，不静默回退。"""
        base = ["--checkpoint", "checkpoint.pth", "--input", "input.png", "--output", "output.png"]
        for bit_mode in BIT_MODES:
            for guidance_mode in ENHANCEMENT_GUIDANCE_MODES:
                flags = ["--bit-mode", bit_mode, "--enhancement-guidance", guidance_mode]
                with patch.object(sys, "argv", ["train.py"] + flags):
                    train_args = training_cli.parse_args()
                infer_args = inference_cli.build_parser().parse_args(base + flags)
                self.assertEqual(train_args.bit_mode, bit_mode)
                self.assertEqual(train_args.enhancement_guidance, guidance_mode)
                self.assertEqual(infer_args.bit_mode, bit_mode)
                self.assertEqual(infer_args.enhancement_guidance, guidance_mode)
        for flag in ("--bit-mode", "--enhancement-guidance"):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()):
                with patch.object(sys, "argv", ["train.py", flag, "unknown"]):
                    with self.assertRaises(SystemExit):
                        training_cli.parse_args()
                with self.assertRaises(SystemExit):
                    inference_cli.build_parser().parse_args(base + [flag, "unknown"])

    def test_inference_uses_selected_model_extractor_and_yaml_cli_precedence(self):
        """实际单图推理使用模型提取器，支持缺省配置、YAML 选择及命令行覆盖。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = directory / "input.png"
            pixels = np.random.default_rng(33).integers(0, 256, (13, 17, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(input_path)
            checkpoint = directory / "model.pth"
            checkpoint_model = SIMI()
            config_path = directory / "experiment.yaml"
            config = self.config()
            config.update(bit_mode="gray", enhancement_guidance="snr")
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            scenarios = (
                ([], ("binary", "none")),
                (["--config", str(config_path)], ("gray", "snr")),
                (["--config", str(config_path), "--bit-mode", "binary", "--enhancement-guidance", "none"], ("binary", "none")),
            )
            for index, (flags, expected_modes) in enumerate(scenarios):
                with self.subTest(scenario=index):
                    captured = []
                    torch.save({
                        "net": checkpoint_model.state_dict(),
                        "model_config": {
                            "bit_mode": expected_modes[0],
                            "enhancement_guidance": expected_modes[1],
                            "enhancement_backbone": "dea",
                        },
                    }, checkpoint)

                    def make_recording_model(*args, **kwargs):
                        bound = inspect.signature(SIMI).bind(*args, **kwargs)
                        bound.apply_defaults()
                        captured.append((bound.arguments["bit_mode"], bound.arguments["enhancement_guidance"]))
                        extractor, network = RecordingExtractor(), RecordingEnhancement()
                        kwargs.update(bitplane_extractor=extractor, enhancement_network=network)
                        model = SIMI(*args, **kwargs)
                        captured.append((model, extractor, network))
                        return model

                    output_path = directory / f"output_{index}.png"
                    arguments = ["--checkpoint", str(checkpoint), "--input", str(input_path), "--output", str(output_path), "--device", "cpu", "--no-benchmark"] + flags
                    namespace = inference_cli.build_parser().parse_args(arguments)
                    with patch.object(inference_cli, "SIMI", side_effect=make_recording_model), redirect_stdout(io.StringIO()):
                        inference_cli.run_inference(namespace)
                    self.assertEqual(captured[0], expected_modes)
                    model, extractor, network = captured[1]
                    self.assertIs(model.bitplane_extractor, extractor)
                    self.assertEqual(len(extractor.inputs), 1)
                    self.assertEqual(len(network.calls), 1)
                    self.assertIs(network.calls[0][1], extractor.outputs[0])
                    self.assertEqual(tuple(extractor.outputs[0].shape), (1, 24, 13, 17))
                    self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
