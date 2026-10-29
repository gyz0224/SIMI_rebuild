"""验证归一化 RGB 的 SSIM 范围与显式旧配置均实际作用于训练。"""

from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from pytorch_msssim import ssim

from losses import UnsupervisedLoss
from train import load_config
from trainers.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]


class LossRangeTests(unittest.TestCase):
    """使用 CPU 小图验证损失数值、梯度及两个去噪训练入口。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def images(self):
        """构造范围为 [0,1]、纹理相反的预测和目标。"""
        generator = torch.Generator().manual_seed(417)
        target = torch.rand((2, 3, 16, 20), generator=generator)
        return 1 - target, target

    def expected_ssim_loss(self, objective, prediction, target, data_range):
        """直接使用同一 SSIM 库核对带权损失，不经过组合损失实现。"""
        return objective.denoising_ssim_weight * torch.mean(
            1 - ssim(prediction, target, data_range=data_range)
        )

    def test_default_range_matches_normalized_rgb_ssim(self):
        """默认范围必须与 data_range=1 的直接调用逐元素一致。"""
        prediction, target = self.images()
        objective = UnsupervisedLoss()
        self.assertEqual(objective.ssim_data_range, 1.0)
        losses = objective.denoising(prediction, target)
        expected = self.expected_ssim_loss(objective, prediction, target, 1.0)
        self.assertTrue(torch.equal(losses["ssim"], expected))
        expected_total = objective.denoising_scale * (
            expected + losses["gradient"] + losses["smoothness"]
        )
        self.assertTrue(torch.equal(losses["total"], expected_total))

    def test_default_yaml_uses_normalized_rgb_range(self):
        """项目默认训练配置也必须使用归一化 RGB 的 SSIM 范围。"""
        config = load_config(ROOT / "configs/default.yaml")
        self.assertEqual(config["loss"]["ssim_data_range"], 1.0)

    def test_explicit_legacy_range_is_preserved_and_materially_different(self):
        """旧范围 255 可显式启用，并与归一化范围产生明显数值差异。"""
        prediction, target = self.images()
        normalized = UnsupervisedLoss()
        legacy = UnsupervisedLoss(ssim_data_range=255.0)
        normalized_losses = normalized.denoising(prediction, target)
        legacy_losses = legacy.denoising(prediction, target)
        expected = self.expected_ssim_loss(legacy, prediction, target, 255.0)
        self.assertTrue(torch.equal(legacy_losses["ssim"], expected))
        self.assertGreater(abs((normalized_losses["ssim"] - legacy_losses["ssim"]).item()), 5.0)
        for name in ("gradient", "smoothness"):
            self.assertTrue(torch.equal(normalized_losses[name], legacy_losses[name]), name)

    def test_both_ranges_have_finite_nonzero_gradients(self):
        """范围修正和显式旧范围均保留有限的反向梯度。"""
        prediction, target = self.images()
        for data_range in (1.0, 255.0):
            with self.subTest(data_range=data_range):
                actual = prediction.clone().requires_grad_()
                losses = UnsupervisedLoss(ssim_data_range=data_range).denoising(actual, target)
                self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
                losses["total"].backward()
                self.assertTrue(torch.isfinite(actual.grad).all())
                self.assertGreater(actual.grad.abs().sum().item(), 0.0)

    def test_joint_and_denoising_training_honor_configured_range(self):
        """joint 与 denoising 的实际训练指标必须采用 YAML 指定的范围。"""
        _, rgb = self.images()
        for mode in ("joint", "denoising"):
            for data_range in (1.0, 255.0):
                with self.subTest(mode=mode, data_range=data_range):
                    config = load_config(ROOT / "configs/default.yaml")
                    config["validation"]["enabled"] = False
                    config["runtime"]["device"] = "cpu"
                    config["train"].update(mode=mode, load_pretrain=False)
                    config["loss"]["ssim_data_range"] = data_range
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(319)
                        trainer = Trainer(config)
                    self.assertEqual(trainer.loss.ssim_data_range, data_range)
                    observed = []
                    original_denoising = trainer.loss.denoising

                    def capture_denoising(final, enhanced):
                        self.assertFalse(enhanced.requires_grad)
                        observed.append((final.detach().clone(), enhanced.detach().clone()))
                        return original_denoising(final, enhanced)

                    with patch.object(trainer.loss, "denoising", side_effect=capture_denoising):
                        metrics = trainer.train_step(rgb)
                    self.assertEqual(len(observed), 1)
                    expected = self.expected_ssim_loss(trainer.loss, *observed[0], data_range)
                    self.assertTrue(torch.equal(metrics["denoising_ssim"], expected))
                    self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
                    self.assertTrue(all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all()
                        for parameter in trainer.model.parameters()
                    ))


if __name__ == "__main__":
    unittest.main()
