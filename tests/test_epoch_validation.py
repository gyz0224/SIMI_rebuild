"""验证每轮独立选优、最新检查点、验证状态续训及带 ETA 的训练日志。"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from train import load_config
from trainers.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]


class RecordingValidator:
    """只替换指标计算，真实训练与检查点保存仍由生产代码执行。"""

    def __init__(self, results):
        self.results = deepcopy(results)
        self.calls = []

    def signature(self):
        """提供固定验证集标识，支持完整续训时的配对数据检查。"""
        return {"pairs": "fixed-test-pair", "lpips_net": "alex"}

    def validate(self, model, mode):
        """记录真实运行模式并消耗随机源，验证训练状态确实被还原。"""
        self.calls.append({
            "mode": mode,
            "model_training": model.training,
            "light_training": model.light_net.training,
            "denoise_training": model.denoise.training,
            "grad_enabled": torch.is_grad_enabled(),
        })
        random.random()
        np.random.random()
        torch.rand(())
        if not self.results:
            raise AssertionError("验证次数超过预设的已完成轮数")
        return self.results.pop(0)


class EpochValidationTests(unittest.TestCase):
    """在 CPU 临时小图上验证工程策略，不运行真实完整验证集。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous_python = random.getstate()
        self.previous_numpy = np.random.get_state()
        self.previous_torch = torch.get_rng_state()
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)

    def tearDown(self):
        random.setstate(self.previous_python)
        np.random.set_state(self.previous_numpy)
        torch.set_rng_state(self.previous_torch)
        torch.use_deterministic_algorithms(self.previous_deterministic)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = self.previous_cudnn

    def config(self, directory, *, epochs=4, enabled=True, count=2):
        """生成两个训练 JPG 和一对验证 PNG，全部路径限定在临时目录。"""
        directory = Path(directory)
        image_root = directory / "images"
        scene = image_root / "scene"
        scene.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(114)
        for index in range(count):
            pixels = generator.integers(0, 256, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(scene / f"{index}.jpg")
        low_dir, high_dir = directory / "validation_low", directory / "validation_high"
        low_dir.mkdir(exist_ok=True)
        high_dir.mkdir(exist_ok=True)
        pixels = np.full((16, 16, 3), 60, dtype=np.uint8)
        Image.fromarray(pixels).save(low_dir / "pair.png")
        Image.fromarray(pixels).save(high_dir / "pair.png")
        config = load_config(ROOT / "configs/default.yaml")
        config["runtime"].update(device="cpu", seed=1143, deterministic=True)
        config["data"].update(root=str(image_root), image_size=16, num_workers=0,
                              shuffle=True, drop_last=False, pin_memory=False, legacy_order=False)
        config["train"].update(mode="enhancement", epochs=epochs, batch_size=2,
                               display_iter=0, snapshots_folder=str(directory / "snapshots"),
                               load_pretrain=False, pretrain_dir="", resume_mode="full",
                               resume_optimizers=True, activation_checkpointing=False)
        # 此组保留旧单验证集配置，专门验证 v2/v3/v4 的兼容行为。
        config["validation"].pop("datasets", None)
        config["validation"].update(enabled=enabled, low_dir=str(low_dir),
                                    high_dir=str(high_dir), lpips_net="alex")
        return config

    def result(self, psnr, ssim, lpips):
        """指标按图像平均，LPIPS 越小越好。"""
        return {"psnr": psnr, "ssim": ssim, "lpips": lpips, "num_images": 1}

    def fit_quietly(self, trainer, **kwargs):
        """仅隐藏测试的训练输出，不替换实际保存动作。"""
        with redirect_stdout(io.StringIO()):
            return trainer.fit(**kwargs)

    def assert_model_equal(self, actual, expected):
        """确保所有参数和缓冲区逐元素一致。"""
        self.assertEqual(list(actual), list(expected))
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_metrics_choose_independent_best_and_ties_do_not_overwrite(self):
        """三个独立最优文件只在严格改善时更新，最新文件始终推进到最后一轮。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            scores = [self.result(10.0, 0.70, 0.4), self.result(12.0, 0.60, 0.2),
                      self.result(11.0, 0.80, 0.3), self.result(12.0, 0.80, 0.2)]
            trainer = Trainer(config)
            trainer.validator = RecordingValidator(scores)
            self.fit_quietly(trainer, max_steps=3)
            folder = Path(config["train"]["snapshots_folder"])
            names = {"best_psnr.pth", "best_ssim.pth", "best_lpips.pth", "latest.pth"}
            self.assertEqual({path.name for path in folder.glob("*.pth")}, names)
            previous_best = {name: (folder / name).read_bytes() for name in names if name != "latest.pth"}
            self.fit_quietly(trainer)
            for name, content in previous_best.items():
                self.assertEqual((folder / name).read_bytes(), content, name)
            self.assertEqual(trainer.best_scores, {"psnr": 12.0, "ssim": 0.80, "lpips": 0.2})
            self.assertEqual(trainer.best_epochs, {"psnr": 1, "ssim": 2, "lpips": 1})
            self.assertEqual([record["epoch"] for record in trainer.validation_history], [0, 1, 2, 3])
            for metric, expected_epoch in (("psnr", 1), ("ssim", 2), ("lpips", 1)):
                payload = torch.load(folder / f"best_{metric}.pth", map_location="cpu", weights_only=True)
                self.assertEqual(payload["training_state"]["epoch"], expected_epoch + 1)
                self.assertEqual(payload["selection_state"]["best_epochs"][metric], expected_epoch)
            latest = torch.load(folder / "latest.pth", map_location="cpu", weights_only=True)
            self.assertEqual(latest["checkpoint_version"], 4)
            self.assertEqual(latest["training_state"]["epoch"], 4)
            self.assertEqual(latest["training_state"]["global_step"], 4)
            self.assertEqual(latest["selection_state"]["validation_history"], trainer.validation_history)
            self.assert_model_equal(latest["net"], trainer.model.state_dict())
            self.assertEqual(len(trainer.validator.calls), 4)

    def test_validation_uses_eval_without_grad_and_restores_training_rng(self):
        """每轮验证进入推理状态，并完整恢复训练分支及三种随机源。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=1)
            trainer = Trainer(deepcopy(config))
            validator = RecordingValidator([self.result(10.0, 0.7, 0.3)])
            trainer.validator = validator
            before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
            observed = trainer.validate_epoch(0)
            self.assertEqual(observed["num_images"], 1)
            self.assertEqual(validator.calls, [{"mode": "enhancement", "model_training": False,
                                               "light_training": False, "denoise_training": False,
                                               "grad_enabled": False}])
            self.assertTrue(trainer.model.light_net.training)
            self.assertFalse(trainer.model.denoise.training)
            self.assertEqual(random.getstate(), before_python)
            after_numpy = np.random.get_state()
            self.assertEqual((after_numpy[0], *after_numpy[2:]), (before_numpy[0], *before_numpy[2:]))
            self.assertTrue(np.array_equal(after_numpy[1], before_numpy[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))

    def test_resume_preserves_best_and_history_then_updates_next_epoch(self):
        """完整续训继承最优分数、历史与计时，不重复验证已完成的轮次。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3)
            first = Trainer(deepcopy(config))
            first.validator = RecordingValidator([self.result(10.0, 0.7, 0.3)])
            self.fit_quietly(first, max_steps=1)
            latest_path = Path(config["train"]["snapshots_folder"]) / "latest.pth"
            saved = torch.load(latest_path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["selection_state"]["best_scores"], first.best_scores)
            resumed_config = deepcopy(config)
            resumed_config["train"].update(load_pretrain=True, pretrain_dir=str(latest_path))
            resumed = Trainer(resumed_config)
            self.assertEqual(resumed.best_scores, first.best_scores)
            self.assertEqual(resumed.best_epochs, first.best_epochs)
            self.assertEqual(resumed.validation_history, first.validation_history)
            resumed.validator = RecordingValidator([self.result(9.0, 0.8, 0.4), self.result(11.0, 0.6, 0.2)])
            self.fit_quietly(resumed)
            self.assertEqual(len(resumed.validator.calls), 2)
            self.assertEqual([record["epoch"] for record in resumed.validation_history], [0, 1, 2])
            self.assertEqual(resumed.best_scores, {"psnr": 11.0, "ssim": 0.8, "lpips": 0.2})
            self.assertEqual(resumed.best_epochs, {"psnr": 2, "ssim": 1, "lpips": 2})
            final = torch.load(latest_path, map_location="cpu", weights_only=True)
            self.assertEqual(final["training_state"]["global_step"], 3)
            self.assertEqual(final["timing_state"]["train_steps"], 3)
            self.assertEqual(final["timing_state"]["epoch_count"], 3)
            self.assertGreaterEqual(final["timing_state"]["train_seconds"], saved["timing_state"]["train_seconds"])

    def test_mid_epoch_stop_saves_latest_without_premature_validation(self):
        """中途停止保存完整续训游标，只有轮次结束才计算指标及保存最佳权重。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=1, count=3)
            trainer = Trainer(config)
            trainer.validator = RecordingValidator([self.result(10.0, 0.7, 0.3)])
            self.fit_quietly(trainer, max_steps=1)
            folder = Path(config["train"]["snapshots_folder"])
            self.assertEqual(sorted(path.name for path in folder.glob("*.pth")), ["latest.pth"])
            self.assertEqual(trainer.validation_history, [])
            self.assertEqual(trainer.validator.calls, [])
            payload = torch.load(folder / "latest.pth", map_location="cpu", weights_only=True)
            self.assertEqual((payload["training_state"]["epoch"], payload["training_state"]["batch_in_epoch"]), (0, 1))
            self.fit_quietly(trainer)
            self.assertEqual(len(trainer.validator.calls), 1)
            self.assertEqual(len(trainer.validation_history), 1)

    def test_weights_policy_resets_metric_selection_and_timing(self):
        """显式开始新阶段时，只加载模型，清除旧阶段最优值、历史和 ETA 累计。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=1)
            source = Trainer(deepcopy(config))
            source.validator = RecordingValidator([self.result(10.0, 0.7, 0.3)])
            self.fit_quietly(source)
            latest = Path(config["train"]["snapshots_folder"]) / "latest.pth"
            receiver = Trainer(deepcopy(config))
            receiver.load_checkpoint(latest, resume_mode="weights")
            self.assertEqual(receiver.best_scores, {"psnr": None, "ssim": None, "lpips": None})
            self.assertEqual(receiver.best_epochs, {"psnr": None, "ssim": None, "lpips": None})
            self.assertEqual(receiver.validation_history, [])
            self.assertEqual(receiver.last_validation, {})
            new_path = receiver.save_checkpoint(Path(directory) / "weights_stage.pth")
            payload = torch.load(new_path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["timing_state"], {"train_seconds": 0.0, "train_steps": 0,
                                                      "epoch_seconds": 0.0, "epoch_count": 0})

    def test_default_logging_every_100_steps_includes_loss_progress_lr_and_eta(self):
        """默认 100 iter 的日志包含全局进度、学习率和可读剩余时间。"""
        self.assertEqual(load_config(ROOT / "configs/default.yaml")["train"]["display_iter"], 100)
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=1, enabled=False, count=101)
            config["train"].update(batch_size=1, display_iter=100)
            trainer = Trainer(config)

            def synthetic_update(rgb):
                """本测试只检查日志频率，避免进行 101 次无关的模型反向计算。"""
                trainer.iteration += 1
                trainer.last_metrics = {"loss_light": torch.tensor(0.25), "loss_exp": torch.tensor(0.125)}
                return trainer.last_metrics

            output = io.StringIO()
            with patch.object(trainer, "train_step", side_effect=synthetic_update), redirect_stdout(output):
                trainer.fit()
            loss_lines = [line for line in output.getvalue().splitlines() if "loss_light:" in line]
            self.assertEqual(len(loss_lines), 1, output.getvalue())
            line = loss_lines[0]
            self.assertIn("Epoch", line)
            self.assertIn("Batch", line)
            self.assertRegex(line, r"Iter\s*[=:]?\s*100/101")
            self.assertIn("lr_light", line)
            self.assertIn("loss_exp:", line)
            self.assertRegex(line, r"ETA\s*=\s*(?:\d+d\s+)?\d{2}:\d{2}:\d{2}")
            self.assertNotIn("nan", line.lower())
            self.assertNotIn("inf", line.lower())

    def test_eta_includes_remaining_updates_and_epoch_validation_cost(self):
        """已知平均耗时下，ETA 同时估计剩余批次和剩余轮末评价耗时。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3, enabled=False, count=5)
            trainer = Trainer(config)
            trainer.build_loader()
            self.assertEqual(trainer.batch_sampler.total_batches, 3)
            trainer.iteration = 4
            trainer.current_epoch = 1
            trainer.timing_state = {"train_seconds": 8.0, "train_steps": 4,
                                    "epoch_seconds": 3.0, "epoch_count": 1}
            # 共九次更新，尚余五次；两次轮末评价各三秒。
            self.assertEqual(trainer._eta_seconds(), 16.0)
            self.assertEqual(trainer._eta_seconds(pending_epoch=True), 19.0)
            trainer.iteration = 9
            trainer.current_epoch = 3
            self.assertEqual(trainer._eta_seconds(), 0.0)
            self.assertEqual(trainer._format_duration(93784), "1d 02:03:04")

    def test_v2_checkpoint_migrates_progress_and_starts_fresh_selection_history(self):
        """原完整续训格式保留数据和优化器进度，新增选优与计时状态从零建立。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=2, enabled=False, count=3)
            source = Trainer(deepcopy(config))
            self.fit_quietly(source, max_steps=1)
            latest = Path(config["train"]["snapshots_folder"]) / "latest.pth"
            old = torch.load(latest, map_location="cpu", weights_only=True)
            old["checkpoint_version"] = 2
            old["training_config"].pop("validation")
            old["training_config"].pop("amp")
            old.pop("amp_scaler")
            old.pop("amp_skipped_steps")
            old.pop("selection_state")
            old.pop("timing_state")
            old_path = Path(directory) / "v2.pth"
            torch.save(old, old_path)
            restored = Trainer(deepcopy(config))
            output = io.StringIO()
            with redirect_stdout(output):
                restored.load_checkpoint(old_path)
            self.assertIn("v2", output.getvalue())
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (0, 1, 1))
            self.assert_model_equal(restored.model.state_dict(), source.model.state_dict())
            self.assertEqual(restored.best_scores, {"psnr": None, "ssim": None, "lpips": None})
            self.assertEqual(restored.validation_history, [])
            self.assertEqual(restored.timing_state, {"train_seconds": 0.0, "train_steps": 0,
                                                    "epoch_seconds": 0.0, "epoch_count": 0})
            self.fit_quietly(restored)
            migrated = torch.load(latest, map_location="cpu", weights_only=True)
            self.assertEqual(migrated["checkpoint_version"], 4)
            self.assertEqual(migrated["training_state"]["global_step"], 4)
            self.assertEqual(migrated["timing_state"]["train_steps"], 3)

    def test_v3_fp32_checkpoint_preserves_selection_and_timing_state(self):
        """旧 FP32 格式迁移时保留选优和计时，缩放器从禁用状态建立。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=2, count=2)
            source = Trainer(deepcopy(config))
            source.validator = RecordingValidator([self.result(12.0, 0.8, 0.2)])
            self.fit_quietly(source, max_steps=1)
            latest = Path(config["train"]["snapshots_folder"]) / "latest.pth"
            old = torch.load(latest, map_location="cpu", weights_only=True)
            old["checkpoint_version"] = 3
            old["training_config"].pop("amp")
            old.pop("amp_scaler")
            old.pop("amp_skipped_steps")
            old_path = latest.with_name("v3.pth")
            torch.save(old, old_path)
            restored = Trainer(deepcopy(config))
            with redirect_stdout(io.StringIO()):
                restored.load_checkpoint(old_path)
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (1, 0, 1))
            self.assert_model_equal(restored.model.state_dict(), source.model.state_dict())
            self.assertEqual(restored.best_scores, source.best_scores)
            self.assertEqual(restored.best_epochs, source.best_epochs)
            self.assertEqual(restored.validation_history, source.validation_history)
            self.assertEqual(restored.timing_state, source.timing_state)
            migrated_path = restored.save_checkpoint(latest.with_name("migrated.pth"))
            migrated = torch.load(migrated_path, map_location="cpu", weights_only=True)
            self.assertEqual(migrated["checkpoint_version"], 4)
            self.assertEqual(migrated["training_config"]["amp"], {"enabled": False})
            self.assertEqual(migrated["amp_scaler"], {})

    def test_full_resume_rejects_changed_validation_dataset_before_updates(self):
        """完整续训改变验证集时提前拒绝，避免不同数据上的历史最佳分数混用。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=2)
            source = Trainer(deepcopy(config))
            source.validator = RecordingValidator([self.result(10.0, 0.7, 0.3)])
            self.fit_quietly(source, max_steps=1)
            checkpoint = Path(config["train"]["snapshots_folder"]) / "latest.pth"
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(checkpoint)
            restored.validator = RecordingValidator([self.result(11.0, 0.8, 0.2)])
            previous_weights = deepcopy(restored.model.state_dict())
            with patch.object(restored.validator, "signature", return_value={"pairs": "changed-pair"}):
                with self.assertRaisesRegex(ValueError, "验证集.*weights"):
                    self.fit_quietly(restored)
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(restored.validator.calls, [])
            self.assert_model_equal(restored.model.state_dict(), previous_weights)
            self.assertEqual(restored.best_scores, source.best_scores)

    def test_manually_confirmed_epoch_boundary_finishes_pending_validation_once(self):
        """轮末游标已经确认但尚未评价的检查点，恢复后补齐一次验证及保存。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=1, count=3)
            source = Trainer(deepcopy(config))
            with redirect_stdout(io.StringIO()):
                loader = source.build_loader()
            for rgb in loader:
                source.train_step(rgb)
                source.batch_sampler.commit()
                source.batch_in_epoch = source.batch_sampler.cursor
            self.assertEqual((source.current_epoch, source.batch_in_epoch, source.iteration), (0, 2, 2))
            saved = source.save_checkpoint(Path(directory) / "confirmed_boundary.pth")
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(saved)
            restored.validator = RecordingValidator([self.result(10.0, 0.7, 0.3)])
            previous_weights = deepcopy(restored.model.state_dict())
            self.fit_quietly(restored)
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (1, 0, 2))
            self.assertEqual(len(restored.validator.calls), 1)
            self.assert_model_equal(restored.model.state_dict(), previous_weights)
            folder = Path(config["train"]["snapshots_folder"])
            self.assertEqual({path.name for path in folder.glob("*.pth")},
                             {"latest.pth", "best_psnr.pth", "best_ssim.pth", "best_lpips.pth"})

    def test_full_resume_to_new_folder_migrates_original_best_weights(self):
        """续训换输出目录后迁移历史最佳权重，后续较差的轮次不能替代它们。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3)
            first = Trainer(deepcopy(config))
            first.validator = RecordingValidator([self.result(12.0, 0.8, 0.2), self.result(10.0, 0.6, 0.4)])
            self.fit_quietly(first, max_steps=2)
            old_folder = Path(config["train"]["snapshots_folder"])
            originals = {name: (old_folder / f"best_{name}.pth").read_bytes()
                         for name in ("psnr", "ssim", "lpips")}
            resumed_config = deepcopy(config)
            new_folder = Path(directory) / "new_output"
            resumed_config["train"].update(snapshots_folder=str(new_folder), load_pretrain=True,
                                            pretrain_dir=str(old_folder / "latest.pth"))
            resumed = Trainer(resumed_config)
            resumed.validator = RecordingValidator([self.result(11.0, 0.7, 0.3)])
            self.fit_quietly(resumed)
            self.assertEqual(resumed.best_epochs, {"psnr": 0, "ssim": 0, "lpips": 0})
            self.assertEqual(resumed.best_scores, {"psnr": 12.0, "ssim": 0.8, "lpips": 0.2})
            for name, contents in originals.items():
                self.assertEqual((new_folder / f"best_{name}.pth").read_bytes(), contents)
                original = torch.load(old_folder / f"best_{name}.pth", map_location="cpu", weights_only=True)
                migrated = torch.load(new_folder / f"best_{name}.pth", map_location="cpu", weights_only=True)
                self.assert_model_equal(migrated["net"], original["net"])
            latest = torch.load(new_folder / "latest.pth", map_location="cpu", weights_only=True)
            self.assertEqual(latest["training_state"]["global_step"], 3)
            self.assertEqual([record["epoch"] for record in resumed.validation_history], [0, 1, 2])

    def test_rolling_back_same_folder_rejects_best_files_from_later_history(self):
        """同目录恢复早期停点时，不能把后续轮次的最佳权重混入旧指标历史。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3)
            source = Trainer(deepcopy(config))
            source.validator = RecordingValidator([self.result(10.0, 0.7, 0.3), self.result(11.0, 0.6, 0.4)])
            self.fit_quietly(source, max_steps=1)
            folder = Path(config["train"]["snapshots_folder"])
            earlier = folder / "earlier_latest.pth"
            shutil.copyfile(folder / "latest.pth", earlier)
            self.fit_quietly(source, max_steps=2)
            latest_contents = (folder / "latest.pth").read_bytes()
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(earlier)
            restored.validator = RecordingValidator([self.result(12.0, 0.8, 0.2)])
            previous_weights = deepcopy(restored.model.state_dict())
            with self.assertRaisesRegex(ValueError, "best_psnr.*历史不一致.*weights"):
                self.fit_quietly(restored)
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(restored.validator.calls, [])
            self.assert_model_equal(restored.model.state_dict(), previous_weights)
            self.assertEqual((folder / "latest.pth").read_bytes(), latest_contents)

    def test_missing_historical_best_rejects_nonbest_latest_as_replacement(self):
        """历史最优文件缺失且当前停点较差时，拒绝伪造最佳模型或部分迁移。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3)
            source = Trainer(deepcopy(config))
            source.validator = RecordingValidator([self.result(12.0, 0.8, 0.2), self.result(10.0, 0.6, 0.4)])
            self.fit_quietly(source, max_steps=2)
            old_folder = Path(config["train"]["snapshots_folder"])
            (old_folder / "best_psnr.pth").unlink()
            resumed_config = deepcopy(config)
            new_folder = Path(directory) / "new_output"
            resumed_config["train"].update(snapshots_folder=str(new_folder), load_pretrain=True,
                                            pretrain_dir=str(old_folder / "latest.pth"))
            restored = Trainer(resumed_config)
            restored.validator = RecordingValidator([self.result(11.0, 0.7, 0.3)])
            previous_weights = deepcopy(restored.model.state_dict())
            with self.assertRaisesRegex(ValueError, "缺少.*best_psnr.*weights"):
                self.fit_quietly(restored)
            self.assertEqual(restored.iteration, 2)
            self.assertEqual(restored.validator.calls, [])
            self.assert_model_equal(restored.model.state_dict(), previous_weights)
            self.assertEqual(list(new_folder.glob("*.pth")), [])


if __name__ == "__main__":
    unittest.main()
