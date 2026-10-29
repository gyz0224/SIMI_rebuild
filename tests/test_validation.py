"""验证每轮有参考评价的真实 LPIPS、阶段选择和训练状态隔离。"""

import copy
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from datasets.bitplane import BinaryBitPlaneExtractor
from trainers.validator import EpochValidator
from utils.image_utils import load_image
from utils.metrics import image_metrics
from utils.reproducibility import capture_rng_state


class _Enhancement(nn.Module):
    """构造含超出 [0,1] 数值的输出，检验评价没有隐藏截断。"""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.5))
        self.normalization = nn.BatchNorm2d(3)
        self.observations = []

    def forward(self, rgb, bitplanes):
        self.observations.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        # 故意消费各随机源，验证结束后必须还原；随机值不影响输出。
        random.random()
        np.random.rand()
        torch.rand(1)
        if rgb.device.type == "cuda":
            torch.rand(1, device=rgb.device)
        assert bitplanes.shape == (rgb.shape[0], 24, *rgb.shape[2:])
        enhanced = rgb * self.gain - 0.2
        return enhanced, rgb, rgb, rgb


class _Denoising(nn.Module):
    """记录去噪调用，不添加新的评价指标或训练逻辑。"""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, rgb):
        self.calls += 1
        return rgb * 0.7


class _StageModel(nn.Module):
    """按 SIMI 的公开输入输出协议提供小型确定性测试模型。"""

    def __init__(self):
        super().__init__()
        self.light_net = _Enhancement()
        self.denoise = _Denoising()
        self.bitplane_extractor = BinaryBitPlaneExtractor()

    def forward(self, rgb, bitplanes):
        enhanced, curve_a, curve_b, _ = self.light_net(rgb, bitplanes)
        return enhanced, curve_a, curve_b, self.denoise(enhanced)


class ValidationTests(unittest.TestCase):
    """使用 64 像素临时 RGB 配对和官方预训练 AlexNet LPIPS。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        cls.low_dir, cls.high_dir = cls.root / "low", cls.root / "high"
        cls.low_dir.mkdir()
        cls.high_dir.mkdir()
        rng = np.random.default_rng(812)
        for index in range(2):
            image = rng.integers(0, 256, (64, 72, 3), dtype=np.uint8)
            target = np.roll(image, index + 1, axis=1)
            Image.fromarray(image).save(cls.low_dir / f"{index}.png")
            Image.fromarray(target).save(cls.high_dir / f"{index}.png")
        cls.config = {
            "enabled": True, "low_dir": str(cls.low_dir),
            "high_dir": str(cls.high_dir), "lpips_net": "alex",
        }
        cls.validator = EpochValidator(cls.config, "cpu")

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def assert_state_equal(self, first, second):
        """比较包含嵌套字典、列表和张量的全部随机状态。"""
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(first, second)

    def expected_metrics(self, validator, mode):
        """直接调用官方指标作为参考，逐图片平均而非拼成一个大 batch。"""
        rows = []
        with torch.inference_mode():
            for source, reference in validator.pairs:
                image = load_image(source).unsqueeze(0).to(validator.device)
                target = load_image(reference).unsqueeze(0).to(validator.device)
                output = image * 1.5 - 0.2
                if mode != "enhancement":
                    output = output * 0.7
                measured = image_metrics(output, target)
                measured["lpips"] = validator.lpips_metric(output * 2 - 1, target * 2 - 1).mean()
                rows.append({key: float(value) for key, value in measured.items()})
        return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}

    def test_lpips_uses_official_pretrained_calibrated_frozen_evaluation(self):
        """检查官方 LPIPS 校准层与特征网络，不允许随机权重伪指标。"""
        metric = self.validator.lpips_metric
        self.assertEqual(metric.version, "0.1")
        self.assertEqual(metric.pnet_type, "alex")
        self.assertTrue(metric.lpips)
        self.assertFalse(metric.pnet_rand)
        self.assertFalse(metric.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in metric.parameters()))
        self.assertEqual(sum(parameter.numel() for parameter in metric.parameters()), 2470848)
        # 核对第一层特征卷积确实等于缓存中的官方 ImageNet 预训练权重。
        from torchvision.models import AlexNet_Weights, alexnet
        rng = capture_rng_state()
        official = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.assertTrue(torch.equal(metric.net.slice1[0].weight, official.features[0].weight))
        from utils.reproducibility import restore_rng_state
        restore_rng_state(rng)

    def test_each_stage_uses_correct_output_and_per_image_mean(self):
        """增强验证跳过去噪；其余阶段验证完整输出，指标均与直接调用一致。"""
        for mode in ("enhancement", "denoising", "joint"):
            with self.subTest(mode=mode):
                model = _StageModel()
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                observed = self.validator.validate(model, mode)
                expected = self.expected_metrics(self.validator, mode)
                self.assertEqual(observed["num_images"], 2)
                for name in ("psnr", "ssim", "lpips"):
                    self.assertAlmostEqual(observed[name], expected[name], places=7)
                self.assertEqual(model.denoise.calls, 0 if mode == "enhancement" else 2)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), parameter_count)
                self.assertTrue(all(observation == (False, False, True) for observation in model.light_net.observations))

    def test_lpips_receives_unclamped_linearly_normalized_rgb(self):
        """保留超范围浮点结果，LPIPS 只做官方线性归一化。"""
        calls = []
        original = self.validator.lpips_metric.forward

        def capture(prediction, target):
            calls.append((prediction.detach().clone(), target.detach().clone()))
            return original(prediction, target)

        with patch.object(self.validator.lpips_metric, "forward", side_effect=capture):
            self.validator.validate(_StageModel(), "enhancement")
        self.assertEqual(len(calls), 2)
        for (source, target_path), (actual, target) in zip(self.validator.pairs, calls):
            image = load_image(source).unsqueeze(0)
            self.assertTrue(torch.equal(actual, (image * 1.5 - 0.2) * 2 - 1))
            self.assertTrue(torch.equal(target, load_image(target_path).unsqueeze(0) * 2 - 1))
            self.assertLess(float(actual.min()), -1)
            self.assertGreater(float(actual.max()), 1)

    def test_validation_restores_nested_training_modes_rng_and_parameters(self):
        """评价结束后保留 Stage 2 的混合 train/eval 状态及所有随机源。"""
        model = _StageModel().train()
        model.light_net.eval()
        model.light_net.normalization.train()
        original_modes = [module.training for module in model.modules()]
        original_parameters = copy.deepcopy(model.state_dict())
        rng = capture_rng_state()
        self.validator.validate(model, "joint")
        self.assert_state_equal(rng, capture_rng_state())
        self.assertEqual(original_modes, [module.training for module in model.modules()])
        self.assert_state_equal(original_parameters, model.state_dict())
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_validation_restores_state_on_failure(self):
        """推理异常不能改变后续训练模式和随机序列。"""
        model = _StageModel().train()
        model.light_net.normalization.eval()
        modes = [module.training for module in model.modules()]
        rng = capture_rng_state()

        def fail(rgb, planes):
            random.random()
            np.random.rand()
            torch.rand(1)
            raise RuntimeError("测试评价失败")

        with patch.object(model, "forward", side_effect=fail), self.assertRaisesRegex(RuntimeError, "测试评价失败"):
            self.validator.validate(model, "joint")
        self.assert_state_equal(rng, capture_rng_state())
        self.assertEqual(modes, [module.training for module in model.modules()])

    def test_constructor_preserves_rng_and_records_metric_definition(self):
        """评价器初始化不改变训练随机状态，指纹记录可复现的指标定义。"""
        rng = capture_rng_state()
        validator = EpochValidator(self.config, "cpu")
        self.assert_state_equal(rng, capture_rng_state())
        self.assertEqual(validator.signature(), self.validator.signature())
        signature = validator.signature()
        self.assertEqual(signature["num_images"], 2)
        self.assertEqual(signature["metrics"]["lpips_version"], "0.1")
        self.assertEqual(signature["metrics"]["average"], "per_image")
        signature["metrics"]["lpips_net"] = "fake"
        self.assertEqual(validator.signature()["metrics"]["lpips_net"], "alex")

    def test_pairing_errors_fail_before_lpips_construction(self):
        """缺失参考、空目录、尺寸不同和同一目录在训练前报告。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            low, high = directory / "low", directory / "high"
            low.mkdir()
            high.mkdir()
            config = {**self.config, "low_dir": str(low), "high_dir": str(high)}
            with patch("lpips.LPIPS") as constructor:
                with self.assertRaisesRegex(ValueError, "为空"):
                    EpochValidator(config, "cpu")
                shutil.copy2(self.low_dir / "0.png", low / "0.png")
                shutil.copy2(self.high_dir / "1.png", high / "1.png")
                with self.assertRaisesRegex(ValueError, "同名配对"):
                    EpochValidator(config, "cpu")
                (high / "1.png").unlink()
                Image.new("RGB", (32, 32)).save(high / "0.png")
                with self.assertRaisesRegex(ValueError, "尺寸不一致"):
                    EpochValidator(config, "cpu")
                with self.assertRaisesRegex(ValueError, "同一个目录"):
                    EpochValidator({**config, "high_dir": str(low)}, "cpu")
                constructor.assert_not_called()

    def test_signature_survives_directory_move_and_detects_changed_image(self):
        """copy2 保留文件属性时目录迁移兼容，图像属性变化必须改变指纹。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            shutil.copytree(self.low_dir, directory / "low", copy_function=shutil.copy2)
            shutil.copytree(self.high_dir, directory / "high", copy_function=shutil.copy2)
            config = {**self.config, "low_dir": str(directory / "low"), "high_dir": str(directory / "high")}
            validator = EpochValidator(config, "cpu")
            self.assertEqual(validator.signature(), self.validator.signature())
            changed = directory / "high" / "0.png"
            Image.new("RGB", (72, 64), color=(10, 20, 30)).save(changed)
            self.assertNotEqual(EpochValidator(config, "cpu").signature(), validator.signature())

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA")
    def test_cuda_metrics_and_rng_are_restored(self):
        """CUDA 指标采用同一真实 LPIPS，验证消耗的 GPU 随机状态可恢复。"""
        torch.cuda.init()
        rng = capture_rng_state()
        validator = EpochValidator(self.config, "cuda:0")
        self.assert_state_equal(rng, capture_rng_state())
        model = _StageModel().to("cuda:0")
        rng = capture_rng_state()
        observed = validator.validate(model, "joint")
        self.assert_state_equal(rng, capture_rng_state())
        expected = self.expected_metrics(validator, "joint")
        for name in expected:
            self.assertAlmostEqual(observed[name], expected[name], places=7)


if __name__ == "__main__":
    unittest.main()
