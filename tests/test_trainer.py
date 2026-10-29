"""验证三种训练模式、位平面入口、检查点和完整配置的生效行为。"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from datasets import BinaryBitPlaneExtractor, bitplanes_from_rgb
from train import load_config
from trainers.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
MODES = ("enhancement", "denoising", "joint")
LIGHT_METRICS = {"loss_light", "loss_smo", "loss_exp", "loss_locl", "loss_gcol"}
NOISE_METRICS = {"loss_noise", "denoising_ssim", "denoising_gradient", "denoising_smoothness"}


class TrainerModeTests(unittest.TestCase):
    """使用 CPU 小批次检查训练行为，不启动正式训练。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, mode="joint"):
        """从完整 YAML 派生测试配置，避免补入隐藏的训练默认值。"""
        config = load_config(ROOT / "configs/default.yaml")
        config["validation"]["enabled"] = False
        config["runtime"]["device"] = "cpu"
        config["train"]["mode"] = mode
        config["train"]["load_pretrain"] = False
        config["train"]["pretrain_dir"] = ""
        config["data"]["num_workers"] = 0
        config["data"]["pin_memory"] = False
        return config

    def trainer(self, config):
        """固定模型初始化，不改变测试结束后的随机状态。"""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1307)
            return Trainer(deepcopy(config))

    def batch(self):
        """生成可供增强和去噪损失共同使用的 16 像素小批次。"""
        generator = torch.Generator().manual_seed(183)
        rgb = torch.rand((2, 3, 16, 16), generator=generator)
        return rgb, bitplanes_from_rgb(rgb)

    def snapshot(self, module):
        """复制完整状态，包含参数和缓冲区。"""
        return {name: value.detach().clone() for name, value in module.state_dict().items()}

    def assert_nested_equal(self, actual, expected):
        """递归比较模型和优化器检查点，张量要求逐元素严格一致。"""
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.shape, expected.shape)
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(list(actual), list(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def assert_branch_state(self, before, module, updated):
        """活动分支必须更新，冻结分支的全部状态必须保持原样。"""
        current = module.state_dict()
        changed = any(not torch.equal(value, current[name]) for name, value in before.items())
        self.assertEqual(changed, updated)
        if not updated:
            self.assert_nested_equal(current, before)

    def test_modes_freeze_branches_and_preserve_gradient_boundaries(self):
        """三种模式只执行需要的分支，并阻断去噪损失对增强参数的梯度。"""
        rgb, planes = self.batch()
        for mode in MODES:
            with self.subTest(mode=mode):
                trainer = self.trainer(self.config(mode))
                light_active = mode != "denoising"
                noise_active = mode != "enhancement"
                self.assertEqual(trainer.mode, mode)
                self.assertEqual(trainer.train_enhancement, light_active)
                self.assertEqual(trainer.train_denoising, noise_active)
                self.assertEqual(trainer.model.light_net.training, light_active)
                self.assertEqual(trainer.model.denoise.training, noise_active)
                self.assertTrue(all(parameter.requires_grad == light_active for parameter in trainer.model.light_net.parameters()))
                self.assertTrue(all(parameter.requires_grad == noise_active for parameter in trainer.model.denoise.parameters()))
                before_light = self.snapshot(trainer.model.light_net)
                before_noise = self.snapshot(trainer.model.denoise)
                light_calls, noise_calls, full_calls, gradients_before_noise = [], [], [], []

                def observe_light(module, inputs, output):
                    light_calls.append((torch.is_grad_enabled(), tuple(value.requires_grad for value in output)))

                def observe_noise(module, inputs):
                    noise_calls.append((torch.is_grad_enabled(), inputs[0].requires_grad, inputs[1].requires_grad))

                original_noise_loss = trainer.loss.denoising

                def observe_noise_loss(final, enhanced):
                    self.assertFalse(enhanced.requires_grad)
                    gradients_before_noise.extend(
                        None if parameter.grad is None else parameter.grad.detach().clone()
                        for parameter in trainer.model.light_net.parameters()
                    )
                    return original_noise_loss(final, enhanced)

                handles = [
                    trainer.model.light_net.register_forward_hook(observe_light),
                    trainer.model.denoise.register_forward_pre_hook(observe_noise),
                    trainer.model.register_forward_hook(lambda module, inputs, output: full_calls.append(output)),
                ]
                try:
                    with patch.object(trainer.loss, "denoising", side_effect=observe_noise_loss):
                        metrics = trainer.train_step(rgb, planes)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(len(light_calls), 1)
                self.assertEqual(light_calls[0], (light_active, (light_active,) * 4))
                self.assertEqual(len(noise_calls), int(noise_active))
                self.assertEqual(len(full_calls), int(mode == "joint"))
                if noise_active:
                    self.assertEqual(noise_calls[0], (True, False, False))
                    for parameter, before in zip(trainer.model.light_net.parameters(), gradients_before_noise):
                        self.assertEqual(parameter.grad is None, before is None)
                        if before is not None:
                            self.assertTrue(torch.equal(parameter.grad, before))
                self.assert_branch_state(before_light, trainer.model.light_net, light_active)
                self.assert_branch_state(before_noise, trainer.model.denoise, noise_active)
                self.assertEqual(any(parameter.grad is not None for parameter in trainer.model.light_net.parameters()), light_active)
                self.assertEqual(any(parameter.grad is not None for parameter in trainer.model.denoise.parameters()), noise_active)
                expected_metrics = (LIGHT_METRICS if light_active else set()) | (NOISE_METRICS if noise_active else set())
                self.assertEqual(set(metrics), expected_metrics)
                self.assertTrue(all(value.ndim == 0 and not value.requires_grad and torch.isfinite(value) for value in metrics.values()))

    def test_only_active_optimizers_and_schedulers_step_in_original_order(self):
        """保持增强先更新、去噪后更新的顺序，并停止冻结分支的调度器。"""
        rgb, planes = self.batch()
        for mode in MODES:
            with self.subTest(mode=mode):
                config = self.config(mode)
                config["scheduler"].update(step_size=1, gamma=0.75)
                trainer = self.trainer(config)
                events = []
                handles = [
                    trainer.optimizer_light.register_step_pre_hook(lambda optimizer, args, kwargs: events.append("light")),
                    trainer.optimizer_denoise.register_step_pre_hook(lambda optimizer, args, kwargs: events.append("noise")),
                ]
                old_epochs = (trainer.scheduler_light.last_epoch, trainer.scheduler_denoise.last_epoch)
                light_step, noise_step = trainer.scheduler_light.step, trainer.scheduler_denoise.step

                def step_light():
                    events.append("scheduler_light")
                    return light_step()

                def step_noise():
                    events.append("scheduler_noise")
                    return noise_step()

                try:
                    with patch.object(trainer.scheduler_light, "step", side_effect=step_light), patch.object(trainer.scheduler_denoise, "step", side_effect=step_noise):
                        trainer.train_step(rgb, planes)
                finally:
                    for handle in handles:
                        handle.remove()
                expected = {
                    "enhancement": ["light", "scheduler_light"],
                    "denoising": ["noise", "scheduler_noise"],
                    "joint": ["light", "noise", "scheduler_light", "scheduler_noise"],
                }
                self.assertEqual(events, expected[mode])
                for index, (optimizer, scheduler, module, active) in enumerate((
                    (trainer.optimizer_light, trainer.scheduler_light, trainer.model.light_net, mode != "denoising"),
                    (trainer.optimizer_denoise, trainer.scheduler_denoise, trainer.model.denoise, mode != "enhancement"),
                )):
                    self.assertEqual([id(parameter) for parameter in optimizer.param_groups[0]["params"]], [id(parameter) for parameter in module.parameters()])
                    self.assertEqual(scheduler.last_epoch, old_epochs[index] + int(active))
                    self.assertAlmostEqual(optimizer.param_groups[0]["lr"], config["train"]["lr"] * (0.75 if active else 1))
                    self.assertEqual(bool(optimizer.state), active)

    def test_internal_bitplanes_match_explicit_bitplanes_for_every_mode(self):
        """训练器设备内拆位与旧显式位平面入口的整个训练步严格等价。"""
        rgb, planes = self.batch()
        for mode in MODES:
            with self.subTest(mode=mode):
                explicit = self.trainer(self.config(mode))
                internal = self.trainer(self.config(mode))
                self.assertIsInstance(internal.bitplane_extractor, BinaryBitPlaneExtractor)
                calls = []
                handle = internal.bitplane_extractor.register_forward_hook(lambda module, inputs, output: calls.append((inputs[0], output)))
                try:
                    expected = explicit.train_step(rgb, planes)
                    actual = internal.train_step(rgb)
                finally:
                    handle.remove()
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0].device, internal.device)
                self.assertTrue(torch.equal(calls[0][1], planes))
                self.assert_nested_equal(actual, expected)
                self.assert_nested_equal(internal.model.state_dict(), explicit.model.state_dict())
                self.assert_nested_equal(internal.optimizer_light.state_dict(), explicit.optimizer_light.state_dict())
                self.assert_nested_equal(internal.optimizer_denoise.state_dict(), explicit.optimizer_denoise.state_dict())

    def test_weights_only_stage_transition_preserves_new_optimizer_configuration(self):
        """增强阶段权重可独立用于去噪阶段，且保留新 YAML 的优化器设置。"""
        rgb, _ = self.batch()
        stage_one = self.trainer(self.config("enhancement"))
        stage_one.train_step(rgb)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"net": stage_one.model.state_dict()}, path)
            config = self.config("denoising")
            config["train"].update(
                load_pretrain=True, pretrain_dir=str(path), resume_mode="weights", resume_optimizers=False,
                lr=0.0003, weight_decay=0.02, adamw_betas=[0.8, 0.95], adamw_eps=1e-6, adamw_amsgrad=True,
            )
            stage_two = self.trainer(config)
            self.assert_nested_equal(stage_two.model.state_dict(), stage_one.model.state_dict())
            for optimizer in (stage_two.optimizer_light, stage_two.optimizer_denoise):
                group = optimizer.param_groups[0]
                self.assertEqual(group["lr"], 0.0003)
                self.assertEqual(group["weight_decay"], 0.02)
                self.assertEqual(tuple(group["betas"]), (0.8, 0.95))
                self.assertEqual(group["eps"], 1e-6)
                self.assertTrue(group["amsgrad"])
                self.assertEqual(len(optimizer.state), 0)
            before_light = self.snapshot(stage_two.model.light_net)
            before_noise = self.snapshot(stage_two.model.denoise)
            stage_two.train_step(rgb)
            self.assert_branch_state(before_light, stage_two.model.light_net, False)
            self.assert_branch_state(before_noise, stage_two.model.denoise, True)

    def test_checkpoint_restore_supports_both_optimizer_policies_and_explicit_override(self):
        """历史 CNN 模式的恢复策略遵从 YAML，显式参数可覆盖并保持旧检查点兼容。"""
        rgb, _ = self.batch()
        source_config = self.config()
        source_config["enhancement_backbone"] = "legacy_cnn"
        source = self.trainer(source_config)
        source.train_step(rgb)
        with tempfile.TemporaryDirectory() as directory:
            path = source.save_checkpoint(Path(directory) / "legacy.pth")
            payload = torch.load(path, map_location="cpu")
            for key in ("net", "optimizer_light", "optimizer_denoise"):
                self.assertIn(key, payload)
            self.assertIn("light_net.e_conv1.weight", payload["net"])
            for configured, override, restore in ((True, None, True), (False, None, False), (True, False, False), (False, True, True)):
                with self.subTest(configured=configured, override=override):
                    config = self.config("denoising")
                    config["enhancement_backbone"] = "legacy_cnn"
                    config["train"].update(resume_mode="legacy", resume_optimizers=configured,
                                           lr=0.0007, weight_decay=0.03)
                    receiver = self.trainer(config)
                    receiver.load_checkpoint(path, resume_optimizers=override)
                    self.assert_nested_equal(receiver.model.state_dict(), source.model.state_dict())
                    for actual, expected in ((receiver.optimizer_light, source.optimizer_light), (receiver.optimizer_denoise, source.optimizer_denoise)):
                        if restore:
                            self.assert_nested_equal(actual.state_dict(), expected.state_dict())
                        else:
                            self.assertEqual(actual.param_groups[0]["lr"], 0.0007)
                            self.assertEqual(actual.param_groups[0]["weight_decay"], 0.03)
                            self.assertEqual(len(actual.state), 0)
            candidates = (ROOT.parent / "SIMI" / "model_new.py", ROOT.parent / "model" / "SIMI" / "model_new.py")
            reference_path = next((path for path in candidates if path.exists()), None)
            if reference_path is not None:
                spec = importlib.util.spec_from_file_location("_trainer_reference", reference_path)
                module = importlib.util.module_from_spec(spec)
                # 即使未使用 -B 启动测试，也不在原项目中生成字节码缓存。
                previous = sys.dont_write_bytecode
                try:
                    sys.dont_write_bytecode = True
                    spec.loader.exec_module(module)
                finally:
                    sys.dont_write_bytecode = previous
                reference = module.bitenhanced()
                reference.load_state_dict(payload["net"], strict=True)
                for name, branch in (("optimizer_light", reference.light_net), ("optimizer_denoise", reference.denoise)):
                    optimizer = torch.optim.AdamW(branch.parameters())
                    optimizer.load_state_dict(payload[name])
                    self.assert_nested_equal(optimizer.state_dict(), payload[name])

    def make_images(self, directory, count=3):
        """只在临时目录生成少量 JPG，验证实际数据加载和检查点落盘。"""
        scene = Path(directory) / "images" / "1"
        scene.mkdir(parents=True)
        generator = np.random.default_rng(771)
        for index in range(count):
            pixels = generator.integers(0, 256, (17, 23, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(scene / f"{index}.jpg")
        return scene.parent

    def test_yaml_hyperparameters_control_training_and_real_data_loading(self):
        """非默认优化器、损失、批次、轮数和数据配置必须实际作用于训练。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config()
            config["data"].update(root=str(self.make_images(directory, count=4)), image_size=20, shuffle=False, pin_memory=False, drop_last=False, num_workers=0)
            config["train"].update(batch_size=2, epochs=2, lr=0.0004, weight_decay=0.012, grad_clip_norm=0.23, display_iter=0,
                                   snapshots_folder=str(Path(directory) / "snapshots"), adamw_betas=[0.7, 0.91], adamw_eps=2e-7, adamw_amsgrad=True)
            config["scheduler"].update(step_size=2, gamma=0.6)
            config["loss"].update(patch_size=5, brightness=0.65, smooth_a_weight=800.0, exposure_weight=2.0)
            trainer = self.trainer(config)
            self.assertEqual(trainer.grad_clip_norm, 0.23)
            self.assertEqual(trainer.loss.smooth_a_weight, 800.0)
            self.assertEqual(trainer.loss.exposure_weight, 2.0)
            self.assertEqual(trainer.loss.exposure.mean_val, 0.65)
            self.assertEqual(trainer.loss.exposure.pool.kernel_size, 5)
            for optimizer in (trainer.optimizer_light, trainer.optimizer_denoise):
                group = optimizer.param_groups[0]
                self.assertEqual(group["lr"], 0.0004)
                self.assertEqual(group["weight_decay"], 0.012)
                self.assertEqual(tuple(group["betas"]), (0.7, 0.91))
                self.assertEqual(group["eps"], 2e-7)
                self.assertTrue(group["amsgrad"])
            with patch("trainers.trainer.DataLoader", wraps=DataLoader) as loader_call, patch.object(trainer, "train_step", wraps=trainer.train_step) as step_call, patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip_call:
                trainer.fit()
            self.assertEqual(step_call.call_count, 4)
            self.assertEqual(trainer.iteration, 4)
            for call in step_call.call_args_list:
                self.assertEqual(tuple(call.args[0].shape), (2, 3, 20, 20))
            options = loader_call.call_args.kwargs
            sampler = options["batch_sampler"]
            self.assertIs(sampler, trainer.batch_sampler)
            self.assertEqual(sampler.batch_size, 2)
            self.assertFalse(sampler.shuffle)
            self.assertFalse(sampler.drop_last)
            self.assertEqual(options["num_workers"], 0)
            self.assertFalse(options["pin_memory"])
            self.assertFalse(loader_call.call_args.args[0].return_bitplanes)
            self.assertEqual(clip_call.call_count, 8)
            self.assertTrue(all(call.args[1] == 0.23 for call in clip_call.call_args_list))
            for optimizer in (trainer.optimizer_light, trainer.optimizer_denoise):
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0004 * 0.6 ** 2)
            folder = Path(config["train"]["snapshots_folder"])
            self.assertEqual(sorted(path.name for path in folder.glob("*.pth")), ["latest.pth"])
            checkpoint = torch.load(folder / "latest.pth", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["training_state"]["epoch"], 2)

    def test_real_jpg_fit_and_checkpoint_for_each_mode(self):
        """三种模式均能读取临时 JPG、重新设置分支状态并保存对应主干格式检查点。"""
        for mode in MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = self.config(mode)
                config["data"].update(root=str(self.make_images(directory)), image_size=16, shuffle=False, drop_last=True)
                config["train"].update(batch_size=2, epochs=1, display_iter=0, snapshots_folder=str(Path(directory) / "snapshots"))
                trainer = self.trainer(config)
                trainer.model.eval()
                metrics = trainer.fit()
                self.assertEqual(trainer.iteration, 1)
                self.assertEqual(trainer.model.light_net.training, mode != "denoising")
                self.assertEqual(trainer.model.denoise.training, mode != "enhancement")
                expected = (LIGHT_METRICS if mode != "denoising" else set()) | (NOISE_METRICS if mode != "enhancement" else set())
                self.assertEqual(set(metrics), expected)
                path = Path(config["train"]["snapshots_folder"]) / "latest.pth"
                self.assertTrue(path.is_file())
                checkpoint = torch.load(path, map_location="cpu")
                self.assertTrue({"net", "optimizer_light", "optimizer_denoise"}.issubset(checkpoint))

    def test_every_yaml_field_is_required_before_model_construction(self):
        """缺少必需字段时在创建模型前报告；关闭验证不要求测试集目录。"""
        complete = self.config()
        for section, values in complete.items():
            keys = values.keys() if isinstance(values, dict) else (None,)
            for key in keys:
                if section == "validation" and key == "datasets" and not values["enabled"]:
                    continue
                with self.subTest(section=section, key=key):
                    config = deepcopy(complete)
                    if key is None:
                        del config[section]
                    else:
                        del config[section][key]
                    with patch("trainers.trainer.SIMI", side_effect=AssertionError("不完整配置不应创建模型")):
                        with self.assertRaises(ValueError):
                            Trainer(config)

    def test_invalid_modes_are_rejected_before_model_construction(self):
        """未支持的模式名称或非字符串模式不能进入训练流程。"""
        for mode in ("all", "stage1", "", None, True, 1):
            with self.subTest(mode=mode):
                config = self.config()
                config["train"]["mode"] = mode
                with patch("trainers.trainer.SIMI", side_effect=AssertionError("非法模式不应创建模型")):
                    with self.assertRaises((ValueError, TypeError)):
                        Trainer(config)


if __name__ == "__main__":
    unittest.main()
