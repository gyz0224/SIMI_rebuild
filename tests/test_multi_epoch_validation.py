"""验证六个测试集各自选优、实时摘要以及完整续训的检查点迁移。"""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
from pathlib import Path
import random
import re
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
DATASET_NAMES = ("LOLv1", "LOLv2real", "LOLv2syn", "LSRWhuawei", "LSRWNikon", "SICE")
METRICS = ("psnr", "ssim", "lpips")


class RecordingMultiValidator:
    """只替换指标计算，训练前向、参数更新和保存仍执行生产代码。"""

    def __init__(self, results):
        self.results = deepcopy(results)
        self.calls = []

    def signature(self):
        """每个测试集具有独立且稳定的配对数据标识。"""
        return {name: {"pairs": f"fixed-pair-{name}", "lpips_net": "alex"}
                for name in DATASET_NAMES}

    def validate(self, model, mode):
        """记录评估模式并扰动随机源，检查验证后是否完整恢复训练状态。"""
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
            raise AssertionError("验证次数超过已完成轮数")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class RecordingSingleValidator(RecordingMultiValidator):
    """模拟旧单测试集配置的 SICE 评价器，保留旧检查点真实保存流程。"""

    def signature(self):
        """旧单集指纹必须仅匹配新多集中的 SICE。"""
        return {"pairs": "fixed-pair-SICE", "lpips_net": "alex"}


class MultiEpochValidationTests(unittest.TestCase):
    """在 CPU 临时小图上检查保存策略，不访问用户权重或真实评估数据。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

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

    def config(self, directory, *, epochs=4, count=2):
        """生成训练 JPG 和六组验证 PNG，所有输出限定在临时 outputs 目录。"""
        directory = Path(directory)
        scene = directory / "images" / "scene"
        scene.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(114)
        for index in range(count):
            pixels = generator.integers(0, 256, (16, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(scene / f"{index}.jpg")
        datasets = {}
        for name in DATASET_NAMES:
            low_dir = directory / "validation" / name / "low"
            high_dir = directory / "validation" / name / "high"
            low_dir.mkdir(parents=True, exist_ok=True)
            high_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((16, 16, 3), 60, dtype=np.uint8)).save(low_dir / "pair.png")
            Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8)).save(high_dir / "pair.png")
            datasets[name] = {"low_dir": str(low_dir), "high_dir": str(high_dir)}
        config = load_config(ROOT / "configs/default.yaml")
        config["runtime"].update(device="cpu", seed=1143, deterministic=True)
        config["data"].update(root=str(scene.parent), image_size=16, num_workers=0,
                              shuffle=True, drop_last=False, pin_memory=False, legacy_order=False)
        config["train"].update(mode="enhancement", epochs=epochs, batch_size=2,
                               display_iter=0, snapshots_folder=str(directory / "outputs" / "weights"),
                               load_pretrain=False, pretrain_dir="", resume_mode="full",
                               resume_optimizers=True, activation_checkpointing=False)
        config["output"] = {"root": str(directory / "outputs")}
        config["validation"] = {"enabled": True, "lpips_net": "alex", "datasets": datasets}
        return config

    @staticmethod
    def score(psnr, ssim, lpips):
        """LPIPS 越小越好；三个指标来自同一轮同一测试集的输出。"""
        return {"psnr": psnr, "ssim": ssim, "lpips": lpips, "num_images": 1}

    def all_scores(self, psnr=10.0, ssim=0.5, lpips=0.5):
        """加入测试集偏移，防止错误地把某一个数据集的结果复制给其他数据集。"""
        return {name: self.score(psnr + index, ssim + index * 0.01, lpips + index * 0.01)
                for index, name in enumerate(DATASET_NAMES)}

    def independent_scores(self):
        """构造三个指标分别在不同轮次和不同测试集改善的结果。"""
        first = self.all_scores()
        second_values = ((12, .4, .6), (9, .7, .6), (9, .4, .3),
                         (12, .7, .3), (9, .4, .6), (12, .4, .3))
        third_values = ((11, .6, .4), (13, .6, .4), (12, .6, .4),
                        (11, .6, .4), (8, .3, .7), (11, .7, .4))
        second, third, ties = {}, {}, {}
        for index, name in enumerate(DATASET_NAMES):
            for destination, values in ((second, second_values), (third, third_values)):
                psnr, ssim, lpips = values[index]
                destination[name] = self.score(psnr + index, ssim + index * .01, lpips + index * .01)
            ties[name] = self.score(max(first[name]["psnr"], second[name]["psnr"], third[name]["psnr"]),
                                    max(first[name]["ssim"], second[name]["ssim"], third[name]["ssim"]),
                                    min(first[name]["lpips"], second[name]["lpips"], third[name]["lpips"]))
        expected_epochs = ((1, 2, 2), (2, 1, 2), (2, 2, 1),
                           (1, 1, 1), (0, 0, 0), (1, 2, 1))
        return [first, second, third, ties], dict(zip(DATASET_NAMES, expected_epochs))

    @staticmethod
    def fit_quietly(trainer, **kwargs):
        """隐藏常规测试日志，同时保持全部实际保存动作。"""
        with redirect_stdout(io.StringIO()):
            return trainer.fit(**kwargs)

    def assert_model_equal(self, actual, expected):
        """参数及缓冲区逐元素相等才视为相同权重。"""
        self.assertEqual(list(actual), list(expected))
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def assert_nested_equal(self, actual, expected):
        """递归比较含张量的优化器、调度器或精度状态，避免张量布尔歧义。"""
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for name in expected:
                self.assert_nested_equal(actual[name], expected[name])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def assert_rng_equal(self, before_python, before_numpy, before_torch):
        """比较 Python、NumPy、PyTorch 三种随机源。"""
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual((after_numpy[0], *after_numpy[2:]), (before_numpy[0], *before_numpy[2:]))
        self.assertTrue(np.array_equal(after_numpy[1], before_numpy[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))

    def assert_summary(self, path, expected_records):
        """摘要必须记录各最佳权重当轮完整三指标，展示轮数从一开始。"""
        text = path.read_text(encoding="utf-8")
        self.assertIn("Best Models Metrics Summary", text)
        for metric, record in expected_records.items():
            marker = f"Best {metric.upper()} Model"
            self.assertIn(marker, text)
            section = text.split(marker, 1)[1]
            section = re.split(r"\n\s*Best (?:PSNR|SSIM|LPIPS) Model", section, maxsplit=1)[0]
            self.assertIn(f"best_{metric}.pth", section)
            epoch = re.search(r"Epoch\s*:\s*(\d+)", section)
            self.assertIsNotNone(epoch, section)
            self.assertEqual(int(epoch.group(1)), record["epoch"] + 1)
            for score in METRICS:
                observed = re.search(rf"{score.upper()}\s*:\s*([+-]?[\d.]+)", section)
                self.assertIsNotNone(observed, section)
                self.assertAlmostEqual(float(observed.group(1)), record[score], places=4)

    def test_six_datasets_choose_independent_best_and_keep_ties(self):
        """十八个最佳权重独立更新，分数相等时权重文件及摘要保持原最佳轮。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            scores, expected_epochs = self.independent_scores()
            trainer = Trainer(config)
            trainer.validator = RecordingMultiValidator(scores)
            self.fit_quietly(trainer, max_steps=3)
            folder = Path(config["train"]["snapshots_folder"])
            original_best = {(name, metric): (folder / name / f"best_{metric}.pth").read_bytes()
                             for name in DATASET_NAMES for metric in METRICS}
            self.fit_quietly(trainer)
            self.assertEqual({path.name for path in folder.glob("*.pth")}, {"latest.pth"})
            self.assertEqual(set(trainer.best_scores), set(DATASET_NAMES))
            for name in DATASET_NAMES:
                self.assertEqual({path.name for path in (folder / name).glob("*.pth")},
                                 {f"best_{metric}.pth" for metric in METRICS})
                for metric, epoch in zip(METRICS, expected_epochs[name]):
                    checkpoint = folder / name / f"best_{metric}.pth"
                    self.assertEqual(checkpoint.read_bytes(), original_best[name, metric])
                    self.assertEqual(trainer.best_epochs[name][metric], epoch)
                    self.assertEqual(trainer.best_scores[name][metric], scores[epoch][name][metric])
                    record = {"epoch": epoch, "global_step": epoch + 1, **scores[epoch][name]}
                    self.assertEqual(trainer.best_records[name][metric], record)
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    self.assertEqual(payload["training_state"]["epoch"], epoch + 1)
                    self.assertEqual(payload["selection_state"]["best_records"][name][metric], record)
                    self.assertEqual(payload["selection_state"]["last_validation"][name], scores[epoch][name])
                self.assert_summary(folder / name / "best_models_metrics.txt", trainer.best_records[name])
            latest = torch.load(folder / "latest.pth", map_location="cpu", weights_only=True)
            self.assertEqual(latest["checkpoint_version"], 5)
            self.assertEqual(latest["training_state"]["global_step"], 4)
            self.assertEqual(latest["selection_state"]["best_scores"], trainer.best_scores)
            self.assertEqual(latest["selection_state"]["best_records"], trainer.best_records)
            self.assertEqual(trainer.validation_history,
                             [{"epoch": epoch, "global_step": epoch + 1, "datasets": scores[epoch]}
                              for epoch in range(4)])
            self.assert_model_equal(latest["net"], trainer.model.state_dict())
            self.assertEqual(len(trainer.validator.calls), 4)

    def test_validation_restores_eval_flags_random_states_and_no_grad(self):
        """六集评价作为一次评估事务执行，恢复所有模块模式和随机源。"""
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(self.config(directory, epochs=1))
            validator = RecordingMultiValidator([self.all_scores()])
            trainer.validator = validator
            modes = {name: module.training for name, module in trainer.model.named_modules()}
            before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
            observed = trainer.validate_epoch(0)
            self.assertEqual(observed, self.all_scores())
            self.assertEqual(validator.calls, [{"mode": "enhancement", "model_training": False,
                                               "light_training": False, "denoise_training": False,
                                               "grad_enabled": False}])
            self.assertEqual({name: module.training for name, module in trainer.model.named_modules()}, modes)
            self.assert_rng_equal(before_python, before_numpy, before_torch)

    def test_invalid_last_dataset_does_not_partially_update_selection(self):
        """最后一个测试集指标非法时，不得先提交前五个测试集的选优结果。"""
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(self.config(directory, epochs=2))
            invalid = self.all_scores(12, .8, .2)
            invalid["SICE"]["ssim"] = float("nan")
            trainer.validator = RecordingMultiValidator([self.all_scores(), invalid])
            trainer.validate_epoch(0)
            previous = deepcopy((trainer.best_scores, trainer.best_epochs, trainer.best_records,
                                 trainer.last_validation, trainer.validation_history))
            before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
            with self.assertRaises((ValueError, RuntimeError)):
                trainer.validate_epoch(1)
            self.assertEqual((trainer.best_scores, trainer.best_epochs, trainer.best_records,
                              trainer.last_validation, trainer.validation_history), previous)
            self.assert_rng_equal(before_python, before_numpy, before_torch)
            self.assertTrue(trainer.model.light_net.training)
            self.assertFalse(trainer.model.denoise.training)

    def test_validator_exception_restores_modes_and_selection(self):
        """评价器异常同样恢复训练状态，并且不生成不完整的验证历史。"""
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(self.config(directory, epochs=1))
            trainer.validator = RecordingMultiValidator([RuntimeError("第六个测试集评估失败")])
            previous = deepcopy((trainer.best_scores, trainer.best_epochs, trainer.best_records,
                                 trainer.last_validation, trainer.validation_history))
            modes = {name: module.training for name, module in trainer.model.named_modules()}
            before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
            with self.assertRaisesRegex(RuntimeError, "第六个测试集"):
                trainer.validate_epoch(0)
            self.assertEqual((trainer.best_scores, trainer.best_epochs, trainer.best_records,
                              trainer.last_validation, trainer.validation_history), previous)
            self.assertEqual({name: module.training for name, module in trainer.model.named_modules()}, modes)
            self.assert_rng_equal(before_python, before_numpy, before_torch)

    def test_invalid_metrics_and_dataset_membership_never_commit_partial_results(self):
        """非有限指标、空集或测试集缺失均整体拒绝，不推进已有最佳记录。"""
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(self.config(directory, epochs=2))
            trainer.validator = RecordingMultiValidator([self.all_scores()])
            trainer.validate_epoch(0)
            previous = deepcopy((trainer.best_scores, trainer.best_epochs, trainer.best_records,
                                 trainer.last_validation, trainer.validation_history))
            invalid_cases = []
            for metric, value in (("psnr", float("nan")), ("psnr", -float("inf")),
                                  ("ssim", float("inf")), ("lpips", float("nan")),
                                  ("lpips", float("inf")), ("num_images", 0), ("num_images", True)):
                result = self.all_scores(12, .8, .2)
                result["SICE"][metric] = value
                invalid_cases.append((f"SICE.{metric}={value}", result))
            missing = self.all_scores(12, .8, .2)
            missing.pop("SICE")
            invalid_cases.append(("缺少SICE", missing))
            extra = self.all_scores(12, .8, .2)
            extra["unknown"] = self.score(12, .8, .2)
            invalid_cases.append(("未知数据集", extra))
            for name, invalid in invalid_cases:
                with self.subTest(case=name):
                    trainer.validator = RecordingMultiValidator([invalid])
                    with self.assertRaises((ValueError, RuntimeError)):
                        trainer.validate_epoch(1)
                    self.assertEqual((trainer.best_scores, trainer.best_epochs, trainer.best_records,
                                      trainer.last_validation, trainer.validation_history), previous)

    def test_mid_epoch_stop_only_saves_latest_and_later_validates_once(self):
        """中途停止不计算六集指标，恢复到轮末后才一次性完成验证。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=1, count=3)
            trainer = Trainer(config)
            trainer.validator = RecordingMultiValidator([self.all_scores()])
            self.fit_quietly(trainer, max_steps=1)
            folder = Path(config["train"]["snapshots_folder"])
            self.assertEqual({path.relative_to(folder).as_posix() for path in folder.rglob("*.pth")},
                             {"latest.pth"})
            self.assertEqual(trainer.validator.calls, [])
            self.assertEqual(trainer.validation_history, [])
            self.fit_quietly(trainer)
            self.assertEqual(len(trainer.validator.calls), 1)
            self.assertEqual(len(list(folder.rglob("best_*.pth"))), 18)
            self.assertEqual(len(list(folder.rglob("best_models_metrics.txt"))), 6)

    def test_full_resume_inherits_nested_selection_and_new_folder_historical_best(self):
        """换输出目录后，十八个真正的历史最优模型原样迁移并重新生成摘要。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3)
            first = Trainer(deepcopy(config))
            first.validator = RecordingMultiValidator([self.all_scores(12, .8, .2), self.all_scores(10, .6, .4)])
            self.fit_quietly(first, max_steps=2)
            old_folder = Path(config["train"]["snapshots_folder"])
            originals = {(name, metric): (old_folder / name / f"best_{metric}.pth").read_bytes()
                         for name in DATASET_NAMES for metric in METRICS}
            resumed_config = deepcopy(config)
            new_root = Path(directory) / "new_outputs"
            new_folder = new_root / "weights"
            resumed_config["output"]["root"] = str(new_root)
            resumed_config["train"].update(snapshots_folder=str(new_folder), load_pretrain=True,
                                            pretrain_dir=str(old_folder / "latest.pth"))
            restored = Trainer(resumed_config)
            self.assertEqual(restored.best_scores, first.best_scores)
            self.assertEqual(restored.best_epochs, first.best_epochs)
            self.assertEqual(restored.best_records, first.best_records)
            self.assertEqual(restored.validation_history, first.validation_history)
            restored.validator = RecordingMultiValidator([self.all_scores(11, .7, .3)])
            self.fit_quietly(restored)
            for name in DATASET_NAMES:
                for metric in METRICS:
                    self.assertEqual((new_folder / name / f"best_{metric}.pth").read_bytes(),
                                     originals[name, metric])
                self.assert_summary(new_folder / name / "best_models_metrics.txt", restored.best_records[name])
            latest = torch.load(new_folder / "latest.pth", map_location="cpu", weights_only=True)
            self.assertEqual(latest["selection_state"]["best_records"], first.best_records)
            self.assertEqual(latest["training_state"]["global_step"], 3)
            self.assertEqual([record["epoch"] for record in restored.validation_history], [0, 1, 2])
            self.assertEqual(len(restored.validator.calls), 1)

    def test_rollback_rejects_later_best_weights_before_any_training_update(self):
        """同目录回滚早期检查点时拒绝后续最佳权重，避免与旧历史混用。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3)
            source = Trainer(deepcopy(config))
            second = self.all_scores()
            second["SICE"]["psnr"] += 2
            source.validator = RecordingMultiValidator([self.all_scores(), second])
            self.fit_quietly(source, max_steps=1)
            folder = Path(config["train"]["snapshots_folder"])
            earlier = folder / "earlier_latest.pth"
            shutil.copyfile(folder / "latest.pth", earlier)
            self.fit_quietly(source, max_steps=2)
            latest_contents = (folder / "latest.pth").read_bytes()
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(earlier)
            restored.validator = RecordingMultiValidator([self.all_scores(13, .9, .1)])
            previous_weights = deepcopy(restored.model.state_dict())
            with self.assertRaisesRegex(ValueError, "best_psnr.*历史不一致.*weights"):
                self.fit_quietly(restored)
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(restored.validator.calls, [])
            self.assert_model_equal(restored.model.state_dict(), previous_weights)
            self.assertEqual((folder / "latest.pth").read_bytes(), latest_contents)

    def test_missing_one_historical_best_rejects_partial_migration_or_latest_replacement(self):
        """最后一个数据集缺失历史最优权重时，不复制前五组文件或伪造最优模型。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3)
            source = Trainer(deepcopy(config))
            source.validator = RecordingMultiValidator([self.all_scores(12, .8, .2), self.all_scores(10, .6, .4)])
            self.fit_quietly(source, max_steps=2)
            old_folder = Path(config["train"]["snapshots_folder"])
            (old_folder / "SICE" / "best_lpips.pth").unlink()
            resumed_config = deepcopy(config)
            new_root = Path(directory) / "new_outputs"
            new_folder = new_root / "weights"
            resumed_config["output"]["root"] = str(new_root)
            resumed_config["train"].update(snapshots_folder=str(new_folder), load_pretrain=True,
                                            pretrain_dir=str(old_folder / "latest.pth"))
            restored = Trainer(resumed_config)
            restored.validator = RecordingMultiValidator([self.all_scores(11, .7, .3)])
            previous_weights = deepcopy(restored.model.state_dict())
            with self.assertRaisesRegex(ValueError, "缺少.*best_lpips.*weights"):
                self.fit_quietly(restored)
            self.assertEqual(restored.iteration, 2)
            self.assertEqual(restored.validator.calls, [])
            self.assert_model_equal(restored.model.state_dict(), previous_weights)
            self.assertEqual(list(new_folder.rglob("*.pth")), [])

    def test_one_changed_dataset_signature_rejects_resume_before_updates(self):
        """任何一个测试集发生变化都拒绝混用历史最佳分数。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=2)
            source = Trainer(deepcopy(config))
            source.validator = RecordingMultiValidator([self.all_scores()])
            self.fit_quietly(source, max_steps=1)
            checkpoint = Path(config["train"]["snapshots_folder"]) / "latest.pth"
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(checkpoint)
            restored.validator = RecordingMultiValidator([self.all_scores(11, .7, .3)])
            changed = restored.validator.signature()
            changed["LSRWNikon"]["pairs"] = "changed-pair"
            previous_weights = deepcopy(restored.model.state_dict())
            with patch.object(restored.validator, "signature", return_value=changed):
                with self.assertRaisesRegex(ValueError, "验证集.*weights"):
                    self.fit_quietly(restored)
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(restored.validator.calls, [])
            self.assert_model_equal(restored.model.state_dict(), previous_weights)

    def test_weights_policy_resets_all_datasets_selection_history_and_records(self):
        """仅加载权重开启新实验时，所有测试集从空最佳记录开始。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=1)
            source = Trainer(deepcopy(config))
            source.validator = RecordingMultiValidator([self.all_scores()])
            self.fit_quietly(source)
            checkpoint = Path(config["train"]["snapshots_folder"]) / "latest.pth"
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(checkpoint, resume_mode="weights")
            for name in DATASET_NAMES:
                self.assertEqual(restored.best_scores[name], {metric: None for metric in METRICS})
                self.assertEqual(restored.best_epochs[name], {metric: None for metric in METRICS})
                self.assertEqual(restored.best_records[name], {metric: None for metric in METRICS})
            self.assertEqual(restored.last_validation, {})
            self.assertEqual(restored.validation_history, [])
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (0, 0, 0))

    def test_dataset_names_cannot_escape_the_weights_directory(self):
        """测试集名用于目录名，必须拒绝路径遍历和平台路径分隔符。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=1)
            for invalid_name in ("../LOLv1", "LOLv1/unsafe", ".", "..", "/tmp/escape", "LOLv1\\unsafe"):
                with self.subTest(name=invalid_name):
                    invalid = deepcopy(config)
                    datasets = invalid["validation"]["datasets"]
                    datasets[invalid_name] = datasets.pop("LOLv1")
                    with self.assertRaises(ValueError):
                        Trainer.validate_config(invalid)

    def test_validation_csv_rebuilt_after_resume_has_no_duplicate_epochs(self):
        """实时指标日志从完整历史重建，续训不会重复写入已完成轮次。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=3)
            source = Trainer(deepcopy(config))
            scores = [self.all_scores(), self.all_scores(11, .7, .3), self.all_scores(12, .8, .2)]
            source.validator = RecordingMultiValidator(scores[:2])
            self.fit_quietly(source, max_steps=2)
            checkpoint = Path(config["train"]["snapshots_folder"]) / "latest.pth"
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(checkpoint)
            restored.validator = RecordingMultiValidator(scores[2:])
            self.fit_quietly(restored)
            path = Path(config["output"]["root"]) / "logs" / "validation_metrics.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 18)
            self.assertEqual({(int(row["epoch"]), row["dataset"]) for row in rows},
                             {(epoch, name) for epoch in (1, 2, 3) for name in DATASET_NAMES})
            for row in rows:
                epoch = int(row["epoch"])
                expected = scores[epoch - 1][row["dataset"]]
                self.assertEqual(int(row["global_step"]), epoch)
                self.assertEqual(int(row["num_images"]), 1)
                for metric in METRICS:
                    self.assertEqual(float(row[metric]), expected[metric])

    def test_legacy_single_sice_full_resume_migrates_only_matching_history(self):
        """旧单集 v4 续训精确保留 SICE 最佳记录，其余五集下轮首次建立记录。"""
        with tempfile.TemporaryDirectory() as directory:
            multi_config = self.config(directory, epochs=3)
            legacy_config = deepcopy(multi_config)
            legacy_config["validation"] = {"enabled": True, "lpips_net": "alex",
                                             **legacy_config["validation"]["datasets"]["SICE"]}
            source = Trainer(legacy_config)
            first_score, second_score = self.score(25, .85, .15), self.score(20, .7, .4)
            source.validator = RecordingSingleValidator([first_score, second_score])
            self.fit_quietly(source, max_steps=2)
            old_folder = Path(legacy_config["train"]["snapshots_folder"])
            latest = old_folder / "latest.pth"
            self.assertEqual(torch.load(latest, map_location="cpu", weights_only=True)["checkpoint_version"], 4)
            originals = {metric: (old_folder / f"best_{metric}.pth").read_bytes() for metric in METRICS}
            new_root = Path(directory) / "multi_outputs"
            new_folder = new_root / "weights"
            multi_config["output"]["root"] = str(new_root)
            multi_config["train"].update(snapshots_folder=str(new_folder), load_pretrain=True,
                                          pretrain_dir=str(latest))
            restored = Trainer(multi_config)
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (2, 0, 2))
            self.assert_nested_equal(restored.optimizer_light.state_dict(), source.optimizer_light.state_dict())
            self.assert_nested_equal(restored.optimizer_denoise.state_dict(), source.optimizer_denoise.state_dict())
            self.assert_nested_equal(restored.scheduler_light.state_dict(), source.scheduler_light.state_dict())
            self.assert_nested_equal(restored.scheduler_denoise.state_dict(), source.scheduler_denoise.state_dict())
            self.assert_nested_equal(restored.scaler.state_dict(), source.scaler.state_dict())
            self.assertEqual(restored.timing_state, source.timing_state)
            new_scores = self.all_scores(11, .7, .3)
            restored.validator = RecordingMultiValidator([new_scores])
            with redirect_stdout(io.StringIO()):
                restored._prepare_validator()
            self.assertEqual(restored.best_scores["SICE"], source.best_scores)
            for name in DATASET_NAMES[:-1]:
                self.assertEqual(restored.best_scores[name], {metric: None for metric in METRICS})
                self.assertEqual(restored.best_records[name], {metric: None for metric in METRICS})
            self.fit_quietly(restored)
            self.assertEqual((restored.current_epoch, restored.iteration), (3, 3))
            self.assertEqual(restored.best_scores["SICE"], {metric: first_score[metric] for metric in METRICS})
            for metric in METRICS:
                self.assertEqual((new_folder / "SICE" / f"best_{metric}.pth").read_bytes(), originals[metric])
                self.assertEqual(restored.best_records["SICE"][metric],
                                 {"epoch": 0, "global_step": 1, **first_score})
            for name in DATASET_NAMES[:-1]:
                self.assertEqual(restored.best_epochs[name], {metric: 2 for metric in METRICS})
                self.assertEqual(restored.best_scores[name], {metric: new_scores[name][metric] for metric in METRICS})
                for metric in METRICS:
                    self.assertTrue((new_folder / name / f"best_{metric}.pth").is_file())
            self.assertEqual(restored.validation_history,
                             [{"epoch": 0, "global_step": 1, "datasets": {"SICE": first_score}},
                              {"epoch": 1, "global_step": 2, "datasets": {"SICE": second_score}},
                              {"epoch": 2, "global_step": 3, "datasets": new_scores}])
            for name in DATASET_NAMES:
                self.assert_summary(new_folder / name / "best_models_metrics.txt", restored.best_records[name])
            migrated = torch.load(new_folder / "latest.pth", map_location="cpu", weights_only=True)
            self.assertEqual(migrated["checkpoint_version"], 5)
            self.assertEqual(migrated["selection_state"]["best_records"], restored.best_records)

    def test_legacy_single_signature_mismatch_rejects_before_best_copy(self):
        """旧数据不能唯一匹配任何当前测试集时，不迁移历史或复制最佳权重。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=2)
            single_config = deepcopy(config)
            single_config["validation"] = {"enabled": True, "lpips_net": "alex",
                                             **single_config["validation"]["datasets"]["SICE"]}
            source = Trainer(single_config)
            source.validator = RecordingSingleValidator([self.score(25, .85, .15)])
            self.fit_quietly(source, max_steps=1)
            checkpoint = Path(single_config["train"]["snapshots_folder"]) / "latest.pth"
            new_root = Path(directory) / "multi_outputs"
            config["output"]["root"] = str(new_root)
            config["train"].update(snapshots_folder=str(new_root / "weights"), load_pretrain=True,
                                    pretrain_dir=str(checkpoint))
            restored = Trainer(config)
            restored.validator = RecordingMultiValidator([self.all_scores()])
            changed = restored.validator.signature()
            changed["SICE"]["pairs"] = "different-pair-SICE"
            previous_weights = deepcopy(restored.model.state_dict())
            with patch.object(restored.validator, "signature", return_value=changed):
                with self.assertRaisesRegex(ValueError, "唯一匹配.*weights"):
                    self.fit_quietly(restored)
            self.assertEqual(restored.iteration, 1)
            self.assertEqual(restored.validator.calls, [])
            self.assert_model_equal(restored.model.state_dict(), previous_weights)
            self.assertEqual(list(new_root.rglob("*.pth")), [])

    def test_legacy_single_direct_save_preserves_pending_migration_without_inference(self):
        """旧单集加载后直接保存，也必须完成历史迁移而不执行评价或训练。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, epochs=2)
            single_config = deepcopy(config)
            single_config["validation"] = {"enabled": True, "lpips_net": "alex",
                                             **single_config["validation"]["datasets"]["SICE"]}
            source = Trainer(single_config)
            score = self.score(25, .85, .15)
            source.validator = RecordingSingleValidator([score])
            self.fit_quietly(source, max_steps=1)
            checkpoint = Path(single_config["train"]["snapshots_folder"]) / "latest.pth"
            restored = Trainer(deepcopy(config))
            restored.load_checkpoint(checkpoint)
            self.assertIsNotNone(restored._legacy_selection_pending)
            previous_weights = deepcopy(restored.model.state_dict())
            previous_optimizers = deepcopy((restored.optimizer_light.state_dict(),
                                            restored.optimizer_denoise.state_dict()))
            previous_modes = {name: module.training for name, module in restored.model.named_modules()}
            before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
            validator = RecordingMultiValidator([])

            def construct_validator(validation_config, device):
                """模拟评价器构造扰动随机源，确认迁移准备过程完整还原随机状态。"""
                random.random()
                np.random.random()
                torch.rand(())
                return validator

            migrated_path = Path(directory) / "direct_migrated.pth"
            with patch("trainers.trainer.MultiDatasetValidator", side_effect=construct_validator) as constructor:
                with redirect_stdout(io.StringIO()):
                    restored.save_checkpoint(migrated_path)
            constructor.assert_called_once()
            self.assertEqual(validator.calls, [])
            self.assertIsNone(restored._legacy_selection_pending)
            self.assertEqual((restored.current_epoch, restored.batch_in_epoch, restored.iteration), (1, 0, 1))
            self.assert_model_equal(restored.model.state_dict(), previous_weights)
            self.assert_nested_equal(restored.optimizer_light.state_dict(), previous_optimizers[0])
            self.assert_nested_equal(restored.optimizer_denoise.state_dict(), previous_optimizers[1])
            self.assertEqual({name: module.training for name, module in restored.model.named_modules()}, previous_modes)
            self.assert_rng_equal(before_python, before_numpy, before_torch)
            migrated = torch.load(migrated_path, map_location="cpu", weights_only=True)
            self.assertEqual(migrated["checkpoint_version"], 5)
            selection = migrated["selection_state"]
            self.assertEqual(selection["best_scores"]["SICE"], {metric: score[metric] for metric in METRICS})
            self.assertEqual(selection["best_epochs"]["SICE"], {metric: 0 for metric in METRICS})
            self.assertEqual(selection["best_records"]["SICE"],
                             {metric: {"epoch": 0, "global_step": 1, **score} for metric in METRICS})
            self.assertEqual(selection["last_validation"], {"SICE": score})
            self.assertEqual(selection["validation_history"],
                             [{"epoch": 0, "global_step": 1, "datasets": {"SICE": score}}])
            self.assertEqual(selection["dataset_signature"], validator.signature())
            for name in DATASET_NAMES[:-1]:
                for field in ("best_scores", "best_epochs", "best_records"):
                    self.assertEqual(selection[field][name], {metric: None for metric in METRICS})
            self.assert_model_equal(migrated["net"], previous_weights)
            self.assert_nested_equal(migrated["optimizer_light"], previous_optimizers[0])
            self.assert_nested_equal(migrated["optimizer_denoise"], previous_optimizers[1])


if __name__ == "__main__":
    unittest.main()
