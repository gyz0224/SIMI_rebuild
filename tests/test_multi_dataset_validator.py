"""验证多数据集独立评价、官方 LPIPS 共享及配对检查的先后顺序。"""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from datasets.bitplane import BinaryBitPlaneExtractor
from trainers.validator import EpochValidator, MultiDatasetValidator
from utils.image_utils import load_image
from utils.metrics import image_metrics
from utils.reproducibility import capture_rng_state


class _PerceptualMetric(nn.Module):
    """替换昂贵感知网络，保留归一化输入与逐图片归约的检查能力。"""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(()))
        self.calls = []

    def forward(self, prediction, target):
        self.calls.append((prediction.detach().clone(), target.detach().clone()))
        return ((prediction - target) ** 2).mean(dim=(1, 2, 3), keepdim=True)


class _Enhancement(nn.Module):
    """记录输入原分辨率、位平面通道数及推理状态。"""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.35))
        self.observations = []

    def forward(self, image, planes):
        self.observations.append((tuple(image.shape), tuple(planes.shape), self.training,
                                  torch.is_grad_enabled()))
        return image * self.gain - 0.15, image, image, image


class _Denoising(nn.Module):
    """记录完整输出路径是否执行了去噪分支。"""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return image * 0.7


class _Model(nn.Module):
    """遵守原 SIMI 输入输出协议的小型确定性模型。"""

    def __init__(self):
        super().__init__()
        self.light_net = _Enhancement()
        self.denoise = _Denoising()
        self.bitplane_extractor = BinaryBitPlaneExtractor()

    def forward(self, image, planes):
        enhanced, a, b, _ = self.light_net(image, planes)
        return enhanced, a, b, self.denoise(enhanced)


class MultiDatasetValidationTests(unittest.TestCase):
    """小型原尺寸图片覆盖配置顺序、各集平均和坏配对早期失败。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, directory, names=("SICE", "LOLv1", "LOLv2real", "LOLv2syn",
                                       "LSRWhuawei", "LSRWNikon")):
        """各集图像数和尺寸不同，避免全局平均错误被相同数据掩盖。"""
        generator = np.random.default_rng(816)
        datasets = {}
        for index, name in enumerate(names):
            low_dir = Path(directory) / name / "low"
            high_dir = Path(directory) / name / "high"
            low_dir.mkdir(parents=True)
            high_dir.mkdir(parents=True)
            for number in range(1 + index % 2):
                pixels = generator.integers(0, 256, (48 + index, 64 + index, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(low_dir / f"{number}.png")
                Image.fromarray(np.roll(pixels, number + 1, axis=1)).save(high_dir / f"{number}.png")
            datasets[name] = {"low_dir": str(low_dir), "high_dir": str(high_dir)}
        return {"enabled": True, "lpips_net": "alex", "datasets": datasets}

    def assert_nested_equal(self, actual, expected):
        """比较包含 Tensor 的随机状态和模型状态。"""
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_six_datasets_share_one_official_frozen_lpips_instance(self):
        """仅此测试构造一次真实 LPIPS，其他测试使用小型感知网络替身。"""
        import lpips

        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            rng = capture_rng_state()
            with patch("lpips.LPIPS", wraps=lpips.LPIPS) as constructor:
                validator = MultiDatasetValidator(config, "cpu")
            constructor.assert_called_once_with(
                net="alex", version="0.1", pretrained=True, pnet_rand=False,
                pnet_tune=False, eval_mode=True, verbose=False,
            )
            metric = validator.lpips_metric
            self.assertFalse(metric.training)
            self.assertFalse(metric.pnet_rand)
            self.assertTrue(metric.lpips)
            self.assertTrue(all(not parameter.requires_grad for parameter in metric.parameters()))
            self.assertEqual(sum(parameter.numel() for parameter in metric.parameters()), 2470848)
            self.assertTrue(all(item.lpips_metric is metric for item in validator.validators.values()))
            self.assertEqual(list(validator.validators), list(config["datasets"]))
            self.assert_nested_equal(capture_rng_state(), rng)

    def test_validation_preserves_order_dimensions_and_independent_per_image_means(self):
        """指标保持原实现，三个阶段使用正确输出，且各集分别按图像平均。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            with patch("lpips.LPIPS", side_effect=lambda **kwargs: _PerceptualMetric()):
                validator = MultiDatasetValidator(config, "cpu")
            for mode in ("enhancement", "denoising", "joint"):
                with self.subTest(mode=mode):
                    model = _Model().train()
                    model.denoise.eval()
                    original_modes = [module.training for module in model.modules()]
                    original_parameters = deepcopy(model.state_dict())
                    rng = capture_rng_state()
                    observed = validator.validate(model, mode)
                    self.assertEqual(list(observed), list(config["datasets"]))
                    expected_shapes = []
                    for name, item in validator.validators.items():
                        rows = []
                        for source, target_path in item.pairs:
                            image = load_image(source).unsqueeze(0)
                            target = load_image(target_path).unsqueeze(0)
                            prediction = image * 1.35 - 0.15
                            if mode != "enhancement":
                                prediction = prediction * 0.7
                            measured = image_metrics(prediction, target)
                            measured["lpips"] = ((prediction * 2 - 1) - (target * 2 - 1)).square().mean()
                            rows.append({key: float(value) for key, value in measured.items()})
                            expected_shapes.append(tuple(image.shape))
                        self.assertEqual(set(observed[name]), {"psnr", "ssim", "lpips", "num_images"})
                        self.assertEqual(observed[name]["num_images"], len(rows))
                        for metric in ("psnr", "ssim", "lpips"):
                            expected = sum(row[metric] for row in rows) / len(rows)
                            self.assertAlmostEqual(observed[name][metric], expected, places=6)
                    self.assertEqual([row[0] for row in model.light_net.observations], expected_shapes)
                    self.assertTrue(all(row[1] == (1, 24, *row[0][2:]) for row in model.light_net.observations))
                    self.assertTrue(all(row[2:] == (False, False) for row in model.light_net.observations))
                    self.assertEqual(model.denoise.calls, 0 if mode == "enhancement" else len(expected_shapes))
                    self.assertEqual([module.training for module in model.modules()], original_modes)
                    self.assert_nested_equal(model.state_dict(), original_parameters)
                    self.assert_nested_equal(capture_rng_state(), rng)
                    self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_bad_later_dataset_fails_before_any_lpips_construction(self):
        """第一集正常、后续集坏配对时，也必须在加载感知网络前报告。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, names=("valid", "broken"))
            high_dir = Path(config["datasets"]["broken"]["high_dir"])
            (high_dir / "1.png").unlink()
            with patch("lpips.LPIPS") as constructor:
                with self.assertRaisesRegex(ValueError, "同名配对"):
                    MultiDatasetValidator(config, "cpu")
                constructor.assert_not_called()

    def test_bad_later_dimensions_fail_before_any_lpips_construction(self):
        """尺寸错误也检查到所有数据集，而不是只检查第一集。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, names=("valid", "broken"))
            target = Path(config["datasets"]["broken"]["high_dir"]) / "0.png"
            Image.new("RGB", (32, 32)).save(target)
            with patch("lpips.LPIPS") as constructor:
                with self.assertRaisesRegex(ValueError, "尺寸不一致"):
                    MultiDatasetValidator(config, "cpu")
                constructor.assert_not_called()

    def test_configuration_rejects_nonmapping_unsafe_names_and_case_duplicates(self):
        """数据集名称不能跨目录，大小写冲突不能覆盖最佳权重目录。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, names=("valid",))
            paths = next(iter(config["datasets"].values()))
            invalid = [[], {}, {"../bad": paths}, {"bad/name": paths}, {"bad\\name": paths},
                       {".": paths}, {"..": paths}, {" ": paths}, {"bad\x00name": paths},
                       {"LOLv1": paths, "lolv1": paths}, {"valid": {}}, {"valid": []}]
            for datasets in invalid:
                with self.subTest(datasets=datasets), patch("lpips.LPIPS") as constructor:
                    with self.assertRaises(ValueError):
                        MultiDatasetValidator({**config, "datasets": datasets}, "cpu")
                    constructor.assert_not_called()

    def test_signature_keeps_dataset_order_original_definition_and_independent_copy(self):
        """多集指纹直接来自单集，不增加新的隐含评价协议。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, names=("second", "first"))
            with patch("lpips.LPIPS", side_effect=lambda **kwargs: _PerceptualMetric()):
                validator = MultiDatasetValidator(config, "cpu")
            signature = validator.signature()
            self.assertEqual(list(signature), ["second", "first"])
            for name, item in validator.validators.items():
                self.assertEqual(signature[name], item.signature())
                self.assertFalse(signature[name]["metrics"]["resize"])
                self.assertFalse(signature[name]["metrics"]["clamp"])
                self.assertEqual(signature[name]["metrics"]["lpips_net"], "alex")
            signature["first"]["metrics"]["resize"] = True
            self.assertFalse(validator.signature()["first"]["metrics"]["resize"])

    def test_epoch_validator_can_receive_shared_metric_without_reconstructing(self):
        """原有两参数调用仍兼容，额外注入时只复用评价器。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, names=("single",))
            single_config = {**config["datasets"]["single"], "lpips_net": "alex"}
            shared = _PerceptualMetric()
            with patch("lpips.LPIPS") as constructor:
                validator = EpochValidator(single_config, "cpu", lpips_model=shared)
            constructor.assert_not_called()
            self.assertIs(validator.lpips_metric, shared)
            self.assertFalse(shared.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in shared.parameters()))

    def test_constructor_restores_random_sources_even_for_shared_initialization(self):
        """共享模型构造消耗的三种随机源不会影响后续训练顺序。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, names=("a", "b"))
            rng = capture_rng_state()

            def construct(**kwargs):
                random.random()
                np.random.random()
                return _PerceptualMetric()

            with patch("lpips.LPIPS", side_effect=construct) as constructor:
                MultiDatasetValidator(config, "cpu")
            constructor.assert_called_once()
            self.assert_nested_equal(capture_rng_state(), rng)


if __name__ == "__main__":
    unittest.main()
