"""使用真实临时 JPG 验证完整续训、采样确认与显式历史恢复策略。"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from datasets import LowLightDataset
from train import load_config
from trainers.trainer import Trainer
from utils.image_utils import load_image


ROOT = Path(__file__).resolve().parents[1]


class TrainingResumeTests(unittest.TestCase):
    """只在 CPU 上运行小批次，验证连续训练与中途恢复严格一致。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous_rng = self.capture_rng()
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    def tearDown(self):
        random.setstate(self.previous_rng["python"])
        numpy_state = self.previous_rng["numpy"]
        np.random.set_state((numpy_state[0], numpy_state[1], *numpy_state[2:]))
        torch.set_rng_state(self.previous_rng["torch"])
        torch.use_deterministic_algorithms(self.previous_deterministic)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = self.previous_cudnn

    def capture_rng(self):
        """独立读取真实随机源，避免仅验证序列化工具自身的输出。"""
        numpy_state = np.random.get_state()
        return {
            "python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
            "torch": torch.get_rng_state().clone(),
        }

    def assert_nested_equal(self, actual, expected, location="state"):
        """递归比较完整状态，张量与 NumPy 数组要求逐元素严格相等。"""
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor, location)
            self.assertEqual(actual.dtype, expected.dtype, location)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape), location)
            self.assertTrue(torch.equal(actual.cpu(), expected.cpu()), location)
        elif isinstance(expected, np.ndarray):
            self.assertTrue(np.array_equal(actual, expected), location)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), location)
            for key, value in expected.items():
                self.assert_nested_equal(actual[key], value, f"{location}.{key}")
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected), location)
            self.assertEqual(len(actual), len(expected), location)
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.assert_nested_equal(left, right, f"{location}[{index}]")
        else:
            self.assertEqual(actual, expected, location)

    def write_image(self, path, index):
        """写入颜色不同的均匀 JPG，使每个样本都可由 RGB 内容唯一识别。"""
        pixels = np.empty((16, 16, 3), dtype=np.uint8)
        pixels[:] = (31 + 23 * index, 55 + 17 * index, 80 + 13 * index)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path, quality=95, subsampling=0)

    def make_dataset(self, directory, count=5):
        """生成一层场景目录，保留不足整批的最后一张图像。"""
        root = Path(directory) / "images"
        for index in range(count):
            self.write_image(root / "scene" / f"{index}.jpg", index)
        return root

    def config(self, root, snapshots, *, mode="joint", workers=0, backbone="dea"):
        """从完整默认 YAML 派生小规模配置，保留所有必填字段。"""
        config = load_config(ROOT / "configs/default.yaml")
        config["validation"]["enabled"] = False
        config["enhancement_backbone"] = backbone
        config["runtime"].update(device="cpu", seed=1143, deterministic=True)
        config["data"].update(root=str(root), image_size=16, num_workers=workers,
                              shuffle=True, pin_memory=False, drop_last=False, legacy_order=False)
        config["train"].update(mode=mode, epochs=2, batch_size=2, display_iter=0,
                               snapshots_folder=str(snapshots), load_pretrain=False, pretrain_dir="",
                               resume_mode="full", resume_optimizers=True,
                               activation_checkpointing=False)
        # 每两次更新调度一次，确保恢复比较确实覆盖已变化的调度器状态。
        config["scheduler"].update(step_size=2, gamma=0.9)
        return config

    def fingerprint(self, rgb):
        """由每张图的实际 RGB 内容识别样本，不依赖采样器内部索引实现。"""
        return tuple(tuple(int(value) for value in torch.round(sample[:, 0, 0] * 255).tolist()) for sample in rgb)

    def record_fit(self, trainer, *, max_steps=None):
        """记录真正训练的批次，并推进三种随机源以验证恢复的连续性。"""
        batches, probes = [], []
        original_step = trainer.train_step

        def recording_step(rgb, *args, **kwargs):
            """只有实际训练批次才记录，工作进程的预取不改变此计数。"""
            batches.append(self.fingerprint(rgb))
            probes.append((random.random(), float(np.random.random()), float(torch.rand(()))))
            return original_step(rgb, *args, **kwargs)

        trainer.train_step = recording_step
        try:
            with redirect_stdout(io.StringIO()):
                trainer.fit(max_steps=max_steps)
        finally:
            trainer.train_step = original_step
        return batches, probes

    def training_snapshot(self, trainer):
        """复制模型、双方优化器、调度器、进度和独立 loader 随机状态。"""
        return deepcopy({
            "model": trainer.model.state_dict(),
            "optimizer_light": trainer.optimizer_light.state_dict(),
            "optimizer_denoise": trainer.optimizer_denoise.state_dict(),
            "scheduler_light": trainer.scheduler_light.state_dict(),
            "scheduler_denoise": trainer.scheduler_denoise.state_dict(),
            "amp_scaler": trainer.scaler.state_dict(),
            "amp_skipped_steps": trainer.amp_skipped_steps,
            "progress": (trainer.current_epoch, trainer.batch_in_epoch, trainer.iteration),
            "sampler": trainer.batch_sampler.state_dict(),
            "loader_generator": trainer.loader_generator.get_state(),
            "rng": self.capture_rng(),
        })

    def test_mid_epoch_full_restore_matches_all_states_and_sample_order(self):
        """三模式及多进程预取下，中途完整恢复与连续两轮训练严格一致。"""
        scenarios = (("enhancement", 0), ("denoising", 0), ("joint", 0), ("joint", 2))
        for mode, workers in scenarios:
            with self.subTest(mode=mode, workers=workers), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                root = self.make_dataset(directory)
                config = self.config(root, directory / "continuous", mode=mode, workers=workers)
                continuous = Trainer(deepcopy(config))
                expected_batches, expected_probes = self.record_fit(continuous)
                expected = self.training_snapshot(continuous)
                self.assertEqual(expected["progress"], (2, 0, 6))
                self.assertEqual([len(batch) for batch in expected_batches], [2, 2, 1, 2, 2, 1])
                self.assertEqual(len({item for batch in expected_batches[:3] for item in batch}), 5)
                self.assertEqual(len({item for batch in expected_batches[3:] for item in batch}), 5)

                config["train"]["snapshots_folder"] = str(directory / "resumed")
                interrupted = Trainer(deepcopy(config))
                initial_batches, initial_probes = self.record_fit(interrupted, max_steps=1)
                self.assertEqual((interrupted.current_epoch, interrupted.batch_in_epoch, interrupted.iteration), (0, 1, 1))
                self.assertEqual(interrupted.batch_sampler.cursor, 1)
                folder = Path(config["train"]["snapshots_folder"])
                self.assertTrue((folder / "latest.pth").is_file())
                self.assertEqual(sorted(path.name for path in folder.glob("*.pth")), ["latest.pth"])

                # 最新停点保留完整数据游标，可通过安全加载模式读取。
                payload = torch.load(folder / "latest.pth", map_location="cpu", weights_only=True)
                self.assertEqual(payload["checkpoint_version"], 4)
                self.assertEqual(payload["training_state"]["global_step"], 1)
                self.assertEqual(payload["training_state"]["epoch"], 0)
                self.assertEqual(payload["training_state"]["batch_in_epoch"], 1)
                resumed_config = deepcopy(config)
                resumed_config["train"].update(load_pretrain=True, pretrain_dir=str(folder / "latest.pth"))
                resumed = Trainer(resumed_config)
                resumed.build_loader()
                self.assertEqual((resumed.current_epoch, resumed.batch_in_epoch, resumed.iteration), (0, 1, 1))
                self.assertEqual(resumed.batch_sampler.cursor, 1)
                stopped_batches, stopped_probes = self.record_fit(resumed, max_steps=1)
                self.assertEqual(stopped_batches, [])
                self.assertEqual(stopped_probes, [])
                remaining_batches, remaining_probes = self.record_fit(resumed)
                self.assertEqual(initial_batches + remaining_batches, expected_batches)
                self.assertEqual(initial_probes + remaining_probes, expected_probes)
                self.assert_nested_equal(self.training_snapshot(resumed), expected)

    def test_prefetch_does_not_advance_acknowledged_sampler_cursor(self):
        """取出或预取批次本身不确认训练完成，游标只能在成功更新后前进。"""
        for workers in (0, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                root = self.make_dataset(directory)
                trainer = Trainer(self.config(root, Path(directory) / "snapshots", workers=workers))
                loader = trainer.build_loader()
                iterator = iter(loader)
                try:
                    rgb = next(iterator)
                    self.assertEqual(rgb.shape[0], 2)
                    self.assertEqual(trainer.batch_sampler.cursor, 0)
                    self.assertEqual((trainer.batch_in_epoch, trainer.iteration), (0, 0))
                    # 消费剩余预取以正常关闭工作进程，整个过程中仍未确认任何更新。
                    list(iterator)
                    self.assertEqual(trainer.batch_sampler.cursor, 0)
                finally:
                    del iterator

    def test_epoch_boundary_restore_updates_latest_without_archived_epoch_files(self):
        """在整轮停点恢复下一轮，最新权重推进且不会重新生成每轮归档。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            config = self.config(root, directory / "snapshots")
            first = Trainer(deepcopy(config))
            self.record_fit(first, max_steps=3)
            self.assertEqual((first.current_epoch, first.batch_in_epoch, first.iteration), (1, 0, 3))
            latest = directory / "snapshots" / "latest.pth"
            original = torch.load(latest, map_location="cpu", weights_only=True)
            self.assertEqual(original["training_state"]["epoch"], 1)
            resumed_config = deepcopy(config)
            resumed_config["train"].update(load_pretrain=True, pretrain_dir=str(latest))
            resumed = Trainer(resumed_config)
            resumed_batches, _ = self.record_fit(resumed)
            self.assertEqual(len(resumed_batches), 3)
            self.assertEqual((resumed.current_epoch, resumed.batch_in_epoch, resumed.iteration), (2, 0, 6))
            final = torch.load(latest, map_location="cpu", weights_only=True)
            self.assertEqual(final["training_state"]["epoch"], 2)
            self.assertEqual(final["training_state"]["global_step"], 6)
            self.assertEqual(sorted(path.name for path in latest.parent.glob("*.pth")), ["latest.pth"])

    def test_natural_completion_updates_latest_checkpoint_to_final_progress(self):
        """正常完成后 latest.pth 记录最终进度，覆盖此前中途停止的旧状态。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            config = self.config(root, directory / "snapshots")
            trainer = Trainer(deepcopy(config))
            self.record_fit(trainer, max_steps=1)
            last = directory / "snapshots" / "latest.pth"
            stopped = torch.load(last, map_location="cpu", weights_only=True)
            self.assertEqual(stopped["training_state"]["global_step"], 1)
            self.record_fit(trainer)
            final = torch.load(last, map_location="cpu", weights_only=True)
            progress = final["training_state"]
            self.assertEqual((progress["epoch"], progress["batch_in_epoch"], progress["global_step"]), (2, 0, 6))
            self.assert_nested_equal(final["net"], trainer.model.state_dict())
            self.assert_nested_equal(final["optimizer_light"], trainer.optimizer_light.state_dict())
            self.assert_nested_equal(final["optimizer_denoise"], trainer.optimizer_denoise.state_dict())
            self.assert_nested_equal(final["scheduler_light"], trainer.scheduler_light.state_dict())
            self.assert_nested_equal(final["scheduler_denoise"], trainer.scheduler_denoise.state_dict())
            # 最终停点再次恢复时，目标总轮数已经达到，不应重复任何训练批次。
            expected = self.training_snapshot(trainer)
            resumed_config = deepcopy(config)
            resumed_config["train"].update(load_pretrain=True, pretrain_dir=str(last))
            resumed = Trainer(resumed_config)
            batches, probes = self.record_fit(resumed)
            self.assertEqual((batches, probes), ([], []))
            self.assert_nested_equal(self.training_snapshot(resumed), expected)

    def test_external_step_checkpoint_requires_weights_policy_for_dataset_fit(self):
        """没有数据游标的检查点可继续外部步骤，切回 fit 必须显式开始新阶段。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            config = self.config(root, directory / "snapshots")
            rgb = load_image(root / "scene" / "0.jpg").unsqueeze(0)
            source = Trainer(deepcopy(config))
            source.train_step(rgb)
            path = source.save_checkpoint(directory / "external_step.pth")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertIsNone(payload["training_state"]["dataset"])
            self.assertIsNone(payload["training_state"]["sampler"])
            self.assertEqual(payload["training_state"]["global_step"], 1)
            source.train_step(rgb)
            expected = deepcopy({
                "net": source.model.state_dict(),
                "optimizer_light": source.optimizer_light.state_dict(),
                "optimizer_denoise": source.optimizer_denoise.state_dict(),
                "scheduler_light": source.scheduler_light.state_dict(),
                "scheduler_denoise": source.scheduler_denoise.state_dict(),
            })

            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(path)
            for operation in (restored.build_loader, lambda: restored.fit(max_steps=2)):
                with self.assertRaisesRegex(ValueError, "train_step.*weights"):
                    operation()
                self.assertEqual(restored.iteration, 1)
                self.assertIsNone(restored.dataset)
            # 被拒绝的是未知数据位置的自动遍历，不影响已明确提供输入的外部更新。
            restored.train_step(rgb)
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (0, 0, 2))
            self.assert_nested_equal(restored.model.state_dict(), expected["net"])
            self.assert_nested_equal(restored.optimizer_light.state_dict(), expected["optimizer_light"])
            self.assert_nested_equal(restored.optimizer_denoise.state_dict(), expected["optimizer_denoise"])
            self.assert_nested_equal(restored.scheduler_light.state_dict(), expected["scheduler_light"])
            self.assert_nested_equal(restored.scheduler_denoise.state_dict(), expected["scheduler_denoise"])

            # 同一个已使用的训练器也必须清除旧动量和进度，随后可重新建立数据加载器。
            restored.load_checkpoint(path, resume_mode="weights")
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (0, 0, 0))
            self.assertEqual(len(restored.optimizer_light.state), 0)
            self.assertEqual(len(restored.optimizer_denoise.state), 0)
            self.assert_nested_equal(restored.model.state_dict(), payload["net"])
            batches, _ = self.record_fit(restored, max_steps=1)
            self.assertEqual(len(batches), 1)
            self.assertEqual(len(batches[0]), 2)
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (0, 1, 1))
            self.assertIsNotNone(restored.dataset)

    def test_external_step_without_sampler_commit_cannot_resume_dataset_fit(self):
        """即使数据清单存在，未确认的外部更新也不能冒充完整的数据进度。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            config = self.config(root, directory / "snapshots")
            source = Trainer(deepcopy(config))
            with redirect_stdout(io.StringIO()):
                loader = source.build_loader()
            iterator = iter(loader)
            rgb = next(iterator)
            del iterator
            source.train_step(rgb)
            self.assertEqual(source.iteration, 1)
            self.assertEqual(source.batch_sampler.cursor, 0)
            self.assertEqual(source.batch_in_epoch, 0)
            path = source.save_checkpoint(directory / "uncommitted_step.pth")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertIsNotNone(payload["training_state"]["dataset"])
            self.assertEqual(payload["training_state"]["sampler"]["cursor"], 0)

            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "global step.*数据进度"):
                restored.fit()
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (0, 0, 1))
            self.assertEqual(restored.batch_sampler.cursor, 0)
            self.assert_nested_equal(restored.model.state_dict(), source.model.state_dict())

    def test_checkpoint_schema_is_safe_and_contains_complete_training_state(self):
        """新检查点无需 pickle 白名单即可读取，包含完整续训所需状态。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            trainer = Trainer(self.config(root, directory / "snapshots"))
            self.record_fit(trainer, max_steps=1)
            payload = torch.load(directory / "snapshots" / "latest.pth", map_location="cpu", weights_only=True)
            self.assertEqual(payload["checkpoint_version"], 4)
            self.assertTrue({"net", "optimizer_light", "optimizer_denoise", "scheduler_light", "scheduler_denoise",
                             "model_config", "training_config", "training_state", "amp_scaler",
                             "amp_skipped_steps"}.issubset(payload))
            # CPU 不启用 CUDA 混合精度，检查点仍明确记录有效精度和空缩放状态。
            self.assertEqual(payload["training_config"]["amp"], {"enabled": False})
            self.assertEqual(payload["amp_scaler"], {})
            self.assertEqual(payload["amp_skipped_steps"], {"light": 0, "denoise": 0})
            self.assertTrue({"epoch", "batch_in_epoch", "global_step", "sampler", "dataset", "rng", "loader_generator"}
                            .issubset(payload["training_state"]))
            self.assert_nested_equal(payload["net"], trainer.model.state_dict())
            self.assert_nested_equal(payload["scheduler_light"], trainer.scheduler_light.state_dict())
            self.assert_nested_equal(payload["scheduler_denoise"], trainer.scheduler_denoise.state_dict())
            self.assert_nested_equal(payload["training_state"]["loader_generator"], trainer.loader_generator.get_state())

    def test_weights_only_stage_transition_starts_fresh_optimizer_scheduler_and_progress(self):
        """增强权重进入去噪新阶段时，不继承旧进度或旧优化器超参数。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            stage_one_config = self.config(root, directory / "stage_one", mode="enhancement")
            stage_one = Trainer(stage_one_config)
            self.record_fit(stage_one, max_steps=1)
            checkpoint = directory / "stage_one" / "latest.pth"
            stage_two_config = self.config(root, directory / "stage_two", mode="denoising")
            stage_two_config["runtime"]["seed"] = 91
            stage_two_config["train"].update(lr=0.0003, weight_decay=0.02, adamw_betas=[0.8, 0.95],
                                              adamw_eps=1e-6, adamw_amsgrad=True, resume_mode="weights",
                                              resume_optimizers=False)
            fresh = Trainer(deepcopy(stage_two_config))
            fresh_rng = self.capture_rng()
            stage_two_config["train"].update(load_pretrain=True, pretrain_dir=str(checkpoint))
            stage_two = Trainer(stage_two_config)
            self.assert_nested_equal(self.capture_rng(), fresh_rng)
            self.assert_nested_equal(stage_two.model.state_dict(), stage_one.model.state_dict())
            self.assertEqual((stage_two.current_epoch, stage_two.batch_in_epoch, stage_two.iteration), (0, 0, 0))
            for optimizer in (stage_two.optimizer_light, stage_two.optimizer_denoise):
                self.assertEqual(len(optimizer.state), 0)
                group = optimizer.param_groups[0]
                self.assertEqual((group["lr"], group["weight_decay"], tuple(group["betas"]), group["eps"], group["amsgrad"]),
                                 (0.0003, 0.02, (0.8, 0.95), 1e-6, True))
            self.assert_nested_equal(stage_two.scheduler_light.state_dict(), fresh.scheduler_light.state_dict())
            self.assert_nested_equal(stage_two.scheduler_denoise.state_dict(), fresh.scheduler_denoise.state_dict())
            before_light = deepcopy(stage_two.model.light_net.state_dict())
            before_denoise = deepcopy(stage_two.model.denoise.state_dict())
            self.record_fit(stage_two, max_steps=1)
            self.assert_nested_equal(stage_two.model.light_net.state_dict(), before_light)
            self.assertTrue(any(not torch.equal(value, stage_two.model.denoise.state_dict()[name]) for name, value in before_denoise.items()))

    def test_historical_three_key_checkpoint_requires_explicit_restore_policy(self):
        """旧三字段检查点不能伪装成完整续训，只能显式加载权重或历史状态。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            config = self.config(root, directory / "source", backbone="legacy_cnn")
            source = Trainer(deepcopy(config))
            source.train_step(load_image(root / "scene" / "0.jpg").unsqueeze(0))
            path = directory / "historical.pth"
            torch.save({"net": source.model.legacy_state_dict(), "optimizer_light": source.optimizer_light.state_dict(),
                        "optimizer_denoise": source.optimizer_denoise.state_dict()}, path)
            receiver = Trainer(deepcopy(config))
            with self.assertRaisesRegex((ValueError, RuntimeError), "weights.*legacy|legacy.*weights"):
                receiver.load_checkpoint(path)
            receiver.load_checkpoint(path, resume_mode="weights")
            self.assert_nested_equal(receiver.model.state_dict(), source.model.state_dict())
            self.assertEqual((receiver.current_epoch, receiver.batch_in_epoch, receiver.iteration), (0, 0, 0))
            self.assertEqual(len(receiver.optimizer_light.state), 0)
            self.assertEqual(len(receiver.optimizer_denoise.state), 0)
            historical = Trainer(deepcopy(config))
            historical.load_checkpoint(path, resume_mode="legacy")
            self.assert_nested_equal(historical.optimizer_light.state_dict(), source.optimizer_light.state_dict())
            self.assert_nested_equal(historical.optimizer_denoise.state_dict(), source.optimizer_denoise.state_dict())
            self.assertEqual((historical.current_epoch, historical.batch_in_epoch, historical.iteration), (0, 0, 0))

    def test_full_restore_rejects_changed_loss_batch_and_dataset_list(self):
        """完整续训拒绝改变 SSIM 范围、batch 划分或数据文件列表。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            config = self.config(root, directory / "snapshots")
            source = Trainer(deepcopy(config))
            self.record_fit(source, max_steps=1)
            checkpoint = directory / "snapshots" / "latest.pth"
            for section, key, value in (("loss", "ssim_data_range", 255.0), ("train", "batch_size", 3)):
                with self.subTest(field=f"{section}.{key}"):
                    changed = deepcopy(config)
                    changed[section][key] = value
                    target = Trainer(changed)
                    with self.assertRaisesRegex((ValueError, RuntimeError), "配置|config|batch|ssim"):
                        target.load_checkpoint(checkpoint)
            self.write_image(root / "scene" / "new.jpg", 5)
            changed_dataset = Trainer(deepcopy(config))
            with self.assertRaisesRegex((ValueError, RuntimeError), "数据|dataset"):
                changed_dataset.load_checkpoint(checkpoint)
                # 文件清单采用延迟加载，在建立训练加载器时必须阻止续训。
                changed_dataset.build_loader()

    def test_same_seed_repeats_initialization_and_order_without_dataset_rng_pollution(self):
        """固定种子独立重建得到相同模型与采样顺序，数据集不改全局 Python RNG。"""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = self.make_dataset(directory)
            random.seed(731)
            before = random.getstate()
            with redirect_stdout(io.StringIO()):
                dataset = LowLightDataset(root, image_size=16, return_bitplanes=False)
            self.assertEqual(random.getstate(), before)
            self.assertEqual(dataset.data_list, sorted(dataset.data_list))
            config = self.config(root, directory / "snapshots")
            first = Trainer(deepcopy(config))
            first_state = deepcopy(first.model.state_dict())
            with redirect_stdout(io.StringIO()):
                first_order = [self.fingerprint(batch) for batch in first.build_loader()]
            second = Trainer(deepcopy(config))
            self.assert_nested_equal(second.model.state_dict(), first_state)
            with redirect_stdout(io.StringIO()):
                second_order = [self.fingerprint(batch) for batch in second.build_loader()]
            self.assertEqual(first_order, second_order)
            self.assertEqual([len(batch) for batch in first_order], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
