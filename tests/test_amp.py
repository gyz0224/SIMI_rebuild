"""验证混合精度训练、梯度缩放边界、溢出保护及完整续训。"""

from contextlib import redirect_stdout
from copy import deepcopy
import gc
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from train import load_config
from trainers.trainer import Trainer
from utils.reproducibility import capture_rng_state, restore_rng_state


ROOT = Path(__file__).resolve().parents[1]


class AMPTestHelpers:
    """用小图像覆盖实际模型更新，不启动数据集正式训练。"""

    def config(self, *, device="cpu", enabled=True, dtype="float16", mode="joint"):
        config = load_config(ROOT / "configs/default.yaml")
        config["validation"]["enabled"] = False
        config["runtime"].update(device=device, seed=1143, deterministic=True)
        # 小图的像素均值梯度较大；使用低初始缩放验证实际更新，而非正常的动态回退。
        config["amp"].update(enabled=enabled, dtype=dtype, init_scale=8.0,
                             growth_factor=2.0, backoff_factor=0.5, growth_interval=2)
        config["data"].update(image_size=16, num_workers=0, pin_memory=False)
        config["train"].update(mode=mode, load_pretrain=False, pretrain_dir="", display_iter=0,
                               activation_checkpointing=False)
        config["scheduler"].update(step_size=1, gamma=0.9)
        return config

    def batch(self, *, index=0):
        """由独立 CPU 随机源生成图像，使测试不改变训练用随机序列。"""
        generator = torch.Generator().manual_seed(2701 + index)
        return torch.rand((1, 3, 16, 16), generator=generator) * 0.6 + 0.05

    def snapshot(self, trainer):
        return deepcopy({
            "model": trainer.model.state_dict(),
            "optimizer_light": trainer.optimizer_light.state_dict(),
            "optimizer_denoise": trainer.optimizer_denoise.state_dict(),
            "scheduler_light": trainer.scheduler_light.state_dict(),
            "scheduler_denoise": trainer.scheduler_denoise.state_dict(),
            "scaler": trainer.scaler.state_dict(),
            "skipped_steps": deepcopy(trainer.amp_skipped_steps),
            "progress": (trainer.current_epoch, trainer.batch_in_epoch, trainer.iteration),
        })

    def assert_nested_equal(self, actual, expected, location="state"):
        """模型、动量和缩放器恢复均要求逐值相等。"""
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype, location)
            self.assertEqual(actual.shape, expected.shape, location)
            self.assertTrue(torch.equal(actual.cpu(), expected.cpu()), location)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), location)
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key], f"{location}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected), location)
            self.assertEqual(len(actual), len(expected), location)
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.assert_nested_equal(left, right, f"{location}[{index}]")
        else:
            self.assertEqual(actual, expected, location)


class AMPConfigTests(AMPTestHelpers, unittest.TestCase):
    """验证 CPU 自动回退、关闭 AMP 的原始行为及配置错误。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous_rng = capture_rng_state()
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    def tearDown(self):
        restore_rng_state(self.previous_rng)
        torch.use_deterministic_algorithms(self.previous_deterministic)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = self.previous_cudnn

    def test_cpu_fallback_matches_explicit_fp32_training_exactly(self):
        """CPU 请求 AMP 仍执行 FP32，模型及双方优化器逐值等于显式关闭 AMP。"""
        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype), redirect_stdout(io.StringIO()):
                fallback = Trainer(self.config(enabled=True, dtype=dtype))
                fp32 = Trainer(self.config(enabled=False, dtype=dtype))
                self.assertFalse(fallback.amp_enabled)
                self.assertFalse(fallback.scaler.is_enabled())
                self.assertFalse(fp32.scaler.is_enabled())
                self.assert_nested_equal(fallback.model.state_dict(), fp32.model.state_dict())
                expected = fp32.train_step(self.batch())
                actual = fallback.train_step(self.batch())
                self.assert_nested_equal(actual, expected)
                self.assert_nested_equal(self.snapshot(fallback), self.snapshot(fp32))
                self.assertTrue(all(value.dtype == torch.float32 for value in actual.values()))

    def test_amp_configuration_rejects_invalid_precision_and_scaler_settings(self):
        """非法精度或缩放参数在训练前明确报错。"""
        cases = (("enabled", "true"), ("dtype", "float32"), ("init_scale", 0.0),
                 ("growth_factor", 1.0), ("backoff_factor", 1.0), ("growth_interval", 0),
                 ("growth_interval", True))
        for key, value in cases:
            with self.subTest(key=key, value=value):
                config = self.config()
                config["amp"][key] = value
                with self.assertRaisesRegex(ValueError, "amp"):
                    Trainer.validate_config(config)

    def test_old_fp32_checkpoint_migration_preserves_training_and_selection_state(self):
        """没有 AMP 信息的 v3 保留优化器、进度及最佳指标，不丢失已有训练。"""
        source = Trainer(self.config(enabled=False))
        source.train_step(self.batch())
        source.best_scores = {"psnr": 18.0, "ssim": 0.65, "lpips": 0.32}
        source.best_epochs = {name: 0 for name in source.best_scores}
        source.validation_history = [{"epoch": 0, "global_step": 1, **source.best_scores, "num_images": 2}]
        with tempfile.TemporaryDirectory() as directory:
            path = source.save_checkpoint(Path(directory) / "historical.pth")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["checkpoint_version"] = 3
            payload["training_config"].pop("amp")
            payload.pop("amp_scaler")
            payload.pop("amp_skipped_steps")
            torch.save(payload, path)
            resumed = Trainer(self.config(enabled=True))
            resumed.load_checkpoint(path)
            self.assert_nested_equal(self.snapshot(resumed), self.snapshot(source))
            self.assertEqual(resumed.best_scores, source.best_scores)
            self.assertEqual(resumed.best_epochs, source.best_epochs)
            self.assertEqual(resumed.validation_history, source.validation_history)
            self.assert_nested_equal(resumed.train_step(self.batch(index=1)), source.train_step(self.batch(index=1)))
            self.assert_nested_equal(self.snapshot(resumed), self.snapshot(source))


@unittest.skipUnless(torch.cuda.is_available(), "AMP CUDA 测试需要可用 GPU")
class AMPCUDATests(AMPTestHelpers, unittest.TestCase):
    """每个批次只有一张 16×16 图像，并限制本测试进程 GPU 分配。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.previous_fraction = torch.cuda.get_per_process_memory_fraction(0)
        # 3090 上上限约为 0.96 GiB，避免干扰用户已经运行的正式训练。
        torch.cuda.set_per_process_memory_fraction(min(cls.previous_fraction, 0.04), 0)

    @classmethod
    def tearDownClass(cls):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(cls.previous_fraction, 0)
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous_rng = capture_rng_state()
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    def tearDown(self):
        restore_rng_state(self.previous_rng)
        torch.use_deterministic_algorithms(self.previous_deterministic)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = self.previous_cudnn
        gc.collect()
        torch.cuda.empty_cache()

    def gpu_config(self, **kwargs):
        return self.config(device="cuda:0", **kwargs)

    def test_feature_prediction_uses_amp_but_curves_and_losses_keep_fp32(self):
        """观察真实 DEA 预测头与损失输入，覆盖检查点重算下的 FP16/BF16。"""
        for dtype in ("float16", "bfloat16"):
            if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                continue
            with self.subTest(dtype=dtype):
                config = self.gpu_config(dtype=dtype)
                config["train"]["activation_checkpointing"] = True
                trainer = Trainer(config)
                expected_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
                heads, curves, noise_inputs, loss_inputs, curve_exponentials = [], [], [], [], []
                before = deepcopy(trainer.model.state_dict())
                original_light_loss, original_noise_loss = trainer.loss.enhancement, trainer.loss.denoising
                original_exp = torch.exp

                def observe_exp(value, *args, **kwargs):
                    curve_exponentials.append((value.dtype, torch.is_autocast_enabled("cuda")))
                    return original_exp(value, *args, **kwargs)

                def observe_light_loss(*args):
                    loss_inputs.append(("light", torch.is_autocast_enabled("cuda"), tuple(x.dtype for x in args)))
                    return original_light_loss(*args)

                def observe_noise_loss(*args):
                    loss_inputs.append(("noise", torch.is_autocast_enabled("cuda"), tuple(x.dtype for x in args)))
                    return original_noise_loss(*args)

                handles = [
                    trainer.model.light_net.curve_network.e_conv1_a.register_forward_hook(
                        lambda module, inputs, output: heads.append(output.dtype)),
                    trainer.model.light_net.register_forward_hook(
                        lambda module, inputs, output: curves.append(tuple(x.dtype for x in output))),
                    trainer.model.denoise.register_forward_pre_hook(
                        lambda module, inputs: noise_inputs.append(tuple(x.requires_grad for x in inputs))),
                ]
                try:
                    with patch.object(trainer.loss, "enhancement", side_effect=observe_light_loss), \
                            patch.object(trainer.loss, "denoising", side_effect=observe_noise_loss), \
                            patch("models.enhancement.torch.exp", side_effect=observe_exp):
                        metrics = trainer.train_step(self.batch())
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertTrue(trainer.amp_enabled)
                self.assertEqual(trainer.scaler.is_enabled(), dtype == "float16")
                self.assertEqual(heads, [expected_dtype])
                # 原始预测 A/B 保留低精度；实际曲线递推及各项 loss 会显式升为 FP32。
                self.assertEqual(curves, [(torch.float32, expected_dtype, expected_dtype, expected_dtype)])
                self.assertGreaterEqual(len(curve_exponentials), 7)
                self.assertEqual(curve_exponentials[:7], [(torch.float32, False)] * 7)
                self.assertEqual(noise_inputs, [(False, False)])
                self.assertEqual(loss_inputs, [
                    ("light", False, (torch.float32,) * 4), ("noise", False, (torch.float32,) * 2),
                ])
                self.assertTrue(all(torch.isfinite(value).item() and value.dtype == torch.float32 for value in metrics.values()))
                self.assertEqual(sum(parameter.numel() for parameter in trainer.model.parameters()), 488044)
                self.assertTrue(all(parameter.dtype == torch.float32 for parameter in trainer.model.parameters()))
                for prefix in ("light_net.", "denoise."):
                    self.assertTrue(any(not torch.equal(value, trainer.model.state_dict()[name])
                                        for name, value in before.items() if name.startswith(prefix)))
                del trainer

    def test_amp_respects_frozen_branch_and_only_steps_active_scheduler(self):
        """分阶段 AMP 仍冻结非活动分支，梯度缩放器不触及其优化器。"""
        for mode in ("enhancement", "denoising"):
            with self.subTest(mode=mode):
                trainer = Trainer(self.gpu_config(mode=mode))
                frozen, active = ((trainer.model.denoise, trainer.model.light_net)
                                  if mode == "enhancement" else (trainer.model.light_net, trainer.model.denoise))
                before_frozen, before_active = deepcopy(frozen.state_dict()), deepcopy(active.state_dict())
                epochs_before = (trainer.scheduler_light.last_epoch, trainer.scheduler_denoise.last_epoch)
                trainer.train_step(self.batch())
                self.assert_nested_equal(frozen.state_dict(), before_frozen)
                self.assertTrue(any(not torch.equal(value, active.state_dict()[name]) for name, value in before_active.items()))
                self.assertEqual(trainer.scheduler_light.last_epoch, epochs_before[0] + int(mode == "enhancement"))
                self.assertEqual(trainer.scheduler_denoise.last_epoch, epochs_before[1] + int(mode == "denoising"))
                del trainer

    def test_unscale_precedes_clipping_and_shared_scaler_updates_once(self):
        """裁剪必须作用于原始梯度，双方更新完成后仅更新一次共享 loss scale。"""
        trainer = Trainer(self.gpu_config())
        events = []
        original_unscale, original_clip = trainer.scaler.unscale_, torch.nn.utils.clip_grad_norm_
        original_update = trainer.scaler.update
        handles = [
            trainer.optimizer_light.register_step_pre_hook(lambda optimizer, args, kwargs: events.append("step_light")),
            trainer.optimizer_denoise.register_step_pre_hook(lambda optimizer, args, kwargs: events.append("step_noise")),
        ]

        def unscale(optimizer):
            events.append("unscale_light" if optimizer is trainer.optimizer_light else "unscale_noise")
            return original_unscale(optimizer)

        def clip(parameters, *args, **kwargs):
            events.append("clip")
            return original_clip(parameters, *args, **kwargs)

        def update(*args, **kwargs):
            events.append("scale_update")
            return original_update(*args, **kwargs)

        try:
            with patch.object(trainer.scaler, "unscale_", side_effect=unscale), \
                    patch.object(trainer.scaler, "update", side_effect=update), \
                    patch("torch.nn.utils.clip_grad_norm_", side_effect=clip):
                trainer.train_step(self.batch())
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(events, ["unscale_light", "clip", "step_light", "unscale_noise", "clip", "step_noise", "scale_update"])

    def test_overflow_skips_only_failed_branch_and_preserves_its_scheduler(self):
        """模拟增强分支 Inf 梯度，去噪仍更新，增强参数及学习率保持原样。"""
        trainer = Trainer(self.gpu_config())
        before_light = deepcopy(trainer.model.light_net.state_dict())
        before_noise = deepcopy(trainer.model.denoise.state_dict())
        epochs_before = (trainer.scheduler_light.last_epoch, trainer.scheduler_denoise.last_epoch)
        scale_before = trainer.scaler.get_scale()
        parameter = next(trainer.model.light_net.parameters())
        handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        try:
            trainer.train_step(self.batch())
        finally:
            handle.remove()
        self.assert_nested_equal(trainer.model.light_net.state_dict(), before_light)
        self.assertTrue(any(not torch.equal(value, trainer.model.denoise.state_dict()[name]) for name, value in before_noise.items()))
        self.assertEqual(trainer.scheduler_light.last_epoch, epochs_before[0])
        self.assertEqual(trainer.scheduler_denoise.last_epoch, epochs_before[1] + 1)
        self.assertEqual(len(trainer.optimizer_light.state), 0)
        self.assertTrue(trainer.optimizer_denoise.state)
        self.assertEqual(trainer.scaler.get_scale(), scale_before * trainer.config["amp"]["backoff_factor"])
        self.assertEqual(trainer.amp_skipped_steps, {"light": 1, "denoise": 0})
        self.assertEqual(trainer.iteration, 1)

    def test_default_scale_recovers_from_natural_overflow_without_nan_parameters(self):
        """默认 scale 在小图梯度溢出后自动降低，随后能真正更新增强参数。"""
        config = self.gpu_config()
        defaults = load_config(ROOT / "configs/default.yaml")["amp"]
        config["amp"].update(init_scale=defaults["init_scale"], growth_interval=defaults["growth_interval"])
        trainer = Trainer(config)
        initial_light = deepcopy(trainer.model.light_net.state_dict())
        initial_scale = trainer.scaler.get_scale()
        metrics = trainer.train_step(self.batch())
        self.assertFalse(trainer.last_step_updates["light"])
        self.assertTrue(trainer.last_step_updates["denoise"])
        self.assertEqual(trainer.scaler.get_scale(), initial_scale * config["amp"]["backoff_factor"])
        for index in range(1, 9):
            self.assertTrue(all(torch.isfinite(value).item() for value in metrics.values()))
            self.assertTrue(all(torch.isfinite(parameter).all().item() for parameter in trainer.model.parameters()))
            if trainer.last_step_updates["light"]:
                break
            metrics = trainer.train_step(self.batch(index=index))
        self.assertTrue(trainer.last_step_updates["light"])
        self.assertTrue(any(not torch.equal(value, trainer.model.light_net.state_dict()[name])
                            for name, value in initial_light.items()))
        self.assertTrue(all(torch.isfinite(parameter).all().item() for parameter in trainer.model.parameters()))
        self.assertGreaterEqual(trainer.amp_skipped_steps["light"], 1)

    def test_checkpoint_restores_scaler_and_next_step_exactly(self):
        """缩放器增长计数、AdamW 动量及下一次 AMP 更新在恢复后逐值一致。"""
        config = self.gpu_config()
        trainer = Trainer(deepcopy(config))
        trainer.train_step(self.batch())
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save_checkpoint(Path(directory) / "amp.pth")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["checkpoint_version"], 4)
            self.assert_nested_equal(payload["amp_scaler"], trainer.scaler.state_dict())
            self.assertTrue(payload["amp_scaler"])
            for enabled, dtype in ((False, "float16"), (True, "bfloat16")):
                if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                    continue
                with self.subTest(changed_precision=(enabled, dtype)):
                    mismatched = Trainer(self.gpu_config(enabled=enabled, dtype=dtype))
                    with self.assertRaisesRegex(ValueError, "AMP|配置"):
                        mismatched.load_checkpoint(path)
                    del mismatched
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(path)
            self.assert_nested_equal(self.snapshot(restored), self.snapshot(trainer))
            expected_metrics = trainer.train_step(self.batch(index=1))
            expected = self.snapshot(trainer)
            actual_metrics = restored.train_step(self.batch(index=1))
            self.assert_nested_equal(actual_metrics, expected_metrics)
            self.assert_nested_equal(self.snapshot(restored), expected)

    def test_weights_only_loading_resets_previous_scaler_history(self):
        """新阶段载入权重时不沿用旧 loss scale 或溢出计数。"""
        trainer = Trainer(self.gpu_config())
        trainer.train_step(self.batch())
        trainer.train_step(self.batch(index=1))
        with tempfile.TemporaryDirectory() as directory:
            path = trainer.save_checkpoint(Path(directory) / "amp.pth")
            receiver = Trainer(self.gpu_config())
            initial_scaler = deepcopy(receiver.scaler.state_dict())
            receiver.train_step(self.batch(index=2))
            receiver.load_checkpoint(path, resume_mode="weights")
            self.assert_nested_equal(receiver.model.state_dict(), trainer.model.state_dict())
            self.assert_nested_equal(receiver.scaler.state_dict(), initial_scaler)
            self.assertEqual(receiver.iteration, 0)
            self.assertEqual(len(receiver.optimizer_light.state), 0)
            self.assertEqual(len(receiver.optimizer_denoise.state), 0)

    def test_epoch_validation_keeps_fp32_inside_external_amp_context(self):
        """即使调用方开启 AMP，最佳权重评价仍使用完整 FP32 推理协议。"""
        config = self.gpu_config()
        # 此用例只检查单评价器内外的精度边界，六集调度另有独立回归。
        config["validation"].pop("datasets", None)
        config["validation"].update(low_dir="unused-low", high_dir="unused-high")
        config["validation"]["enabled"] = True
        trainer = Trainer(config)
        observed = {"heads": [], "outputs": [], "autocast": [], "inference": []}
        rgb = self.batch().to(trainer.device)

        class FP32Validator:
            """实际执行模型，只替换指标计算，以隔离推理精度边界的检查。"""

            def signature(self):
                return {"protocol": "fp32-test"}

            def validate(self, model, mode):
                observed["autocast"].append(torch.is_autocast_enabled("cuda"))
                observed["inference"].append(torch.is_inference_mode_enabled())
                observed["outputs"].append(tuple(value.dtype for value in model(rgb)))
                return {"psnr": 18.0, "ssim": 0.65, "lpips": 0.32, "num_images": 1}

        trainer.validator = FP32Validator()
        handle = trainer.model.light_net.curve_network.e_conv1_a.register_forward_hook(
            lambda module, inputs, output: observed["heads"].append(output.dtype))
        modes_before = [(module, module.training) for module in trainer.model.modules()]
        scaler_before = deepcopy(trainer.scaler.state_dict())
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                scores = trainer.validate_epoch(0)
                self.assertTrue(torch.is_autocast_enabled("cuda"))
        finally:
            handle.remove()
        self.assertEqual(observed, {
            "heads": [torch.float32], "outputs": [(torch.float32,) * 4],
            "autocast": [False], "inference": [True],
        })
        self.assertEqual(scores, {"psnr": 18.0, "ssim": 0.65, "lpips": 0.32, "num_images": 1})
        self.assertEqual(trainer.best_scores, {"psnr": 18.0, "ssim": 0.65, "lpips": 0.32})
        self.assertEqual([(module, module.training) for module in trainer.model.modules()], modes_before)
        self.assert_nested_equal(trainer.scaler.state_dict(), scaler_before)


if __name__ == "__main__":
    unittest.main()
