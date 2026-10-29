"""将重构实现与相邻目录中未修改的原始 SIMI 源码进行对照。"""

import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import TensorDataset
from pytorch_msssim import ssim

from datasets import LowLightDataset, bitplanes_from_rgb, decompose_bitplanes
from losses import UnsupervisedLoss
from models import SIMI


ROOT = Path(__file__).resolve().parents[1]
# 同时兼容原始项目位置和归档位置，避免目录整理后跳过等价性验证。
SOURCE = next(
    (path for path in (ROOT.parent / "SIMI", ROOT.parent / "model" / "SIMI")
     if (path / "model_new.py").is_file()),
    ROOT.parent / "SIMI",
)


def import_source(filename):
    spec = importlib.util.spec_from_file_location("_reference_" + filename[:-3], SOURCE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bitplanes_all_byte_values(self):
        image = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16).expand(3, -1, -1)
        expected = np.unpackbits(image.permute(1, 2, 0).numpy(), axis=2)
        expected = torch.from_numpy(expected.transpose(2, 0, 1).copy()).float()
        self.assertTrue(torch.equal(decompose_bitplanes(image), expected))
        batch = torch.stack([image, 255 - image])
        planes = decompose_bitplanes(batch)
        self.assertEqual(tuple(planes.shape), (2, 24, 16, 16))
        self.assertTrue(torch.equal(bitplanes_from_rgb(batch.float() / 255), planes))

    def test_parameter_counts_and_rgb_interface(self):
        model = SIMI(enhancement_backbone="legacy_cnn")
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        self.assertEqual(count(model.light_net), 105907)
        self.assertEqual(count(model.denoise), 110312)
        self.assertEqual(count(model), 216219)
        rgb = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            outputs = model(rgb)
            self.assertEqual(len(outputs), 4)
            self.assertTrue(torch.equal(model.enhance(rgb), outputs[3]))
            features = model.extract_features(rgb)
        self.assertIs(features["rgb"], rgb)
        self.assertEqual(tuple(features["raw_bitplanes"].shape), (1, 24, 16, 16))
        self.assertEqual(tuple(features["bit_plane_features"].shape), (1, 24, 16, 16))
        self.assertEqual(tuple(features["fused_features"].shape), (1, 27, 16, 16))

    def test_strict_checkpoint_contract(self):
        model = SIMI(enhancement_backbone="legacy_cnn")
        clone = SIMI(enhancement_backbone="legacy_cnn")
        clone.load_state_dict(model.state_dict())
        clone.load_state_dict(model.legacy_state_dict())
        malformed = model.legacy_state_dict()
        malformed.pop(next(iter(malformed)))
        with self.assertRaises(RuntimeError):
            clone.load_legacy_state_dict(malformed)
        malformed = model.legacy_state_dict()
        key = next(iter(malformed))
        malformed[key] = malformed[key][:1]
        with self.assertRaises(RuntimeError):
            clone.load_legacy_state_dict(malformed)


@unittest.skipUnless((SOURCE / "model_new.py").exists(), "Original sibling SIMI is required for reference comparisons")
class LegacyEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference_model = import_source("model_new.py")
        cls.reference_loss = import_source("Myloss.py")
        cls.reference_data = import_source("dataloader_new.py")

    def pair(self):
        torch.manual_seed(2026)
        legacy = self.reference_model.bitenhanced()
        torch.manual_seed(2026)
        rebuilt = SIMI(enhancement_backbone="legacy_cnn")
        return legacy, rebuilt

    def assert_state_equal(self, actual, expected):
        self.assertEqual(list(actual), list(expected))
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]), key)

    def test_initialization_and_parameter_order(self):
        legacy, rebuilt = self.pair()
        self.assert_state_equal(rebuilt.legacy_state_dict(), legacy.state_dict())
        for old, new in zip(legacy.parameters(), rebuilt.parameters()):
            self.assertEqual(old.shape, new.shape)
            self.assertTrue(torch.equal(old, new))

    def test_forward_and_gradients(self):
        legacy, rebuilt = self.pair()
        rgb = torch.rand(2, 3, 16, 16)
        planes = bitplanes_from_rgb(rgb)
        old_outputs = legacy(rgb, planes)
        new_outputs = rebuilt(rgb, planes)
        for old, new in zip(old_outputs, new_outputs):
            self.assertTrue(torch.equal(old, new))
        sum(output.sum() for output in old_outputs).backward()
        sum(output.sum() for output in new_outputs).backward()
        for old, new in zip(legacy.parameters(), rebuilt.parameters()):
            self.assertEqual(old.grad is None, new.grad is None)
            if old.grad is not None:
                self.assertTrue(torch.equal(old.grad, new.grad))
        self.assertIsNone(rebuilt.light_net.curve_network.e_conv1_b.weight.grad)
        self.assertIsNone(rebuilt.denoise.batch_norm.weight.grad)

    def test_denoising_detach_boundary(self):
        model = SIMI(enhancement_backbone="legacy_cnn")
        rgb = torch.rand(1, 3, 16, 16)
        model(rgb)[3].sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in model.light_net.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in model.denoise.parameters()))

    def test_dataset_matches_original(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "1"
            scene.mkdir()
            generator = np.random.default_rng(11)
            for index in range(3):
                image = generator.integers(0, 256, (23, 31, 3), dtype=np.uint8)
                Image.fromarray(image).save(scene / f"{index}.jpg")
            random.seed(1143)
            original = self.reference_data.lowlight_loader(directory)
            original.size = 32
            random.seed(1143)
            rebuilt = LowLightDataset(directory, image_size=32, legacy_order=True)
            self.assertEqual(original.data_list, rebuilt.data_list)
            for index in range(len(original)):
                for old, new in zip(original[index], rebuilt[index]):
                    self.assertTrue(torch.equal(old, new))

    def test_loss_values_and_gradients(self):
        # 旧源码隐式使用 SSIM 库的 255 范围，等价性对照必须显式选择旧配置。
        loss = UnsupervisedLoss(ssim_data_range=255.0)
        original = torch.rand(2, 3, 16, 16)
        old_inputs = [torch.rand(2, channels, 16, 16, requires_grad=True) for channels in (3, 21, 21)]
        new_inputs = [tensor.detach().clone().requires_grad_() for tensor in old_inputs]
        enhanced, a, b = old_inputs
        smooth = self.reference_loss.L_SMO()
        expected = {
            "smoothness": 1000 * smooth(a) + 5000 * smooth(b),
            "exposure": 5 * self.reference_loss.L_exp(4, 0.8)(enhanced, original).mean(),
            "local_color": 1000 * self.reference_loss.L_localcolor()(enhanced, original).mean(),
            "global_color": 1500 * self.reference_loss.L_color()(enhanced).mean(),
        }
        expected["total"] = expected["exposure"] + expected["smoothness"] + expected["local_color"] + expected["global_color"]
        actual = loss.enhancement(new_inputs[0], original, new_inputs[1], new_inputs[2])
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)
        expected["total"].backward()
        actual["total"].backward()
        for old, new in zip(old_inputs, new_inputs):
            self.assertTrue(torch.equal(old.grad, new.grad))
        target = enhanced.detach()
        old_final = torch.rand_like(target, requires_grad=True)
        new_final = old_final.detach().clone().requires_grad_()
        old_noise = 10 * (10 * torch.mean(1 - ssim(old_final, target)) + 40 * torch.mean(self.reference_loss.L_gradient()(old_final, target)) + 1 * torch.mean(smooth(old_final)))
        new_noise = loss.denoising(new_final, target)["total"]
        self.assertTrue(torch.equal(old_noise, new_noise))
        old_noise.backward()
        new_noise.backward()
        self.assertTrue(torch.equal(old_final.grad, new_final.grad))

    def test_actual_legacy_training_step(self):
        """使用一个 CPU 合成数据批次运行未修改的原始训练函数。"""
        from train import load_config
        from trainers.trainer import Trainer

        sys.path.insert(0, str(SOURCE))
        try:
            source_train = import_source("train_new.py")
        finally:
            sys.path.pop(0)
        legacy, rebuilt = self.pair()
        rgb = torch.rand(1, 3, 16, 16)
        planes = bitplanes_from_rgb(rgb)
        config = load_config(ROOT / "configs/default.yaml")
        config["validation"]["enabled"] = False
        # 原源码历史回归使用 FP32，避免默认 AMP 设置影响等价性检验。
        config.setdefault("amp", {})["enabled"] = False
        config["enhancement_backbone"] = "legacy_cnn"
        config["loss"]["ssim_data_range"] = 255.0
        config["runtime"]["device"] = "cpu"
        # 旧数学回归显式保留外部随机种子、旧列表顺序与旧检查点恢复语义。
        config["runtime"].update(seed=None, deterministic=False)
        config["data"]["legacy_order"] = True
        config["train"]["resume_mode"] = "legacy"
        trainer = Trainer(config)
        trainer.model.load_state_dict(rebuilt.state_dict())
        optimizers = []
        adamw = torch.optim.AdamW

        def capture_optimizer(*args, **kwargs):
            optimizer = adamw(*args, **kwargs)
            optimizers.append(optimizer)
            return optimizer

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                lr=1e-5, weight_decay=1e-4, load_pretrain=False,
                lowlight_images_path=directory, train_batch_size=1, num_workers=0,
                num_epochs=1, grad_clip_norm=0.1, display_iter=100,
                snapshots_folder=directory + "/",
            )
            with patch.object(source_train.model, "bitenhanced", return_value=legacy), \
                 patch.object(source_train.dataloader, "lowlight_loader", return_value=TensorDataset(rgb, planes)), \
                 patch.object(torch.nn.Module, "cuda", lambda self, *args, **kwargs: self), \
                 patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self), \
                 patch.object(torch.optim, "AdamW", side_effect=capture_optimizer), \
                 patch.dict(source_train.os.environ, {}, clear=False):
                source_train.train(args)
            trainer.train_step(rgb, planes)
            self.assert_state_equal(trainer.model.legacy_state_dict(), legacy.state_dict())
            for old, new in zip(optimizers, (trainer.optimizer_light, trainer.optimizer_denoise)):
                old_state = old.state_dict()
                new_state = new.state_dict()
                self.assertEqual(old_state["param_groups"], new_state["param_groups"])
                self.assertEqual(old_state["state"].keys(), new_state["state"].keys())
                for index, state in old_state["state"].items():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            self.assertTrue(torch.equal(value, new_state["state"][index][key]))
                        else:
                            self.assertEqual(value, new_state["state"][index][key])
            restored = Trainer(config)
            restored.load_checkpoint(Path(directory) / "Epoch0.pth")
            self.assert_state_equal(restored.model.legacy_state_dict(), legacy.state_dict())


if __name__ == "__main__":
    unittest.main()
