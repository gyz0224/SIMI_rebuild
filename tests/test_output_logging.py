"""验证输出目录覆盖、日志追加、错误捕获及标准输出流恢复。"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch

import train
from utils.logging import tee_training_log, validate_output_paths


class FakeTrainer:
    """替代耗时模型更新，仅验证训练入口的配置与日志生命周期。"""

    MODES = ("enhancement", "denoising", "joint")
    received_configs = []

    @staticmethod
    def validate_config(config):
        """模拟训练前的配置检查，确保非法覆盖项先于日志创建被拒绝。"""
        if config["train"]["batch_size"] < 1:
            raise ValueError("train.batch_size 必须为正整数")

    def __init__(self, config):
        self.received_configs.append(deepcopy(config))
        self.device = torch.device("cpu")
        self.amp_enabled = False
        self.amp_dtype_name = "float16"
        self.scaler = type("FakeScaler", (), {"is_enabled": lambda self: False})()
        print("模型初始化日志", flush=True)
        print("评价器初始化警告", file=sys.stderr, flush=True)

    def fit(self, max_steps=None):
        """输出正常训练与评价日志，不创建权重或执行模型计算。"""
        print(f"训练调用 max_steps={max_steps}", flush=True)
        print("逐轮评价日志", file=sys.stderr, flush=True)


class OutputLoggingTests(unittest.TestCase):
    """所有文件都写入临时目录，避免接触用户正式训练输出。"""

    def config(self, directory):
        """提供命令行所需的配置字段，真实模型由 FakeTrainer 隔离。"""
        output_root = Path(directory) / "outputs"
        return {
            "output": {"root": str(output_root)},
            "runtime": {"device": "cpu", "seed": 1143},
            "amp": {"enabled": True, "dtype": "float16"},
            "train": {"batch_size": 8, "epochs": 500, "snapshots_folder": str(output_root / "weights")},
        }

    def run_main(self, config, arguments=()):
        """保留实际参数解析和文件写入，仅替换配置读取与训练计算。"""
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(train, "load_config", return_value=deepcopy(config)), \
                patch.object(train, "Trainer", FakeTrainer), \
                patch.object(sys, "argv", ["train.py", *arguments]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            train.main()
        return stdout.getvalue(), stderr.getvalue()

    def test_tee_captures_stdout_stderr_and_warnings_without_hiding_console(self):
        """正常输出和 warnings 同时出现在各自终端流与同一日志文件。"""
        with tempfile.TemporaryDirectory() as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr), warnings.catch_warnings():
                warnings.simplefilter("always")
                with tee_training_log(directory) as path:
                    print("训练输出", flush=True)
                    print("标准错误", file=sys.stderr, flush=True)
                    warnings.warn("LPIPS 初始化提示", UserWarning)
                    # 上下文内即可读到文本，证明日志无需等待训练结束才刷新。
                    current = path.read_text(encoding="utf-8")
                    self.assertIn("训练输出", current)
                    self.assertIn("LPIPS 初始化提示", current)
            content = path.read_text(encoding="utf-8")
            self.assertIn("训练输出", stdout.getvalue())
            self.assertIn("标准错误", stderr.getvalue())
            self.assertIn("LPIPS 初始化提示", stderr.getvalue())
            self.assertIn("标准错误", content)

    def test_log_appends_and_restores_both_streams(self):
        """第二次运行保留第一次记录，并恢复上下文进入前的输出流。"""
        with tempfile.TemporaryDirectory() as directory:
            before_stdout, before_stderr = sys.stdout, sys.stderr
            console = io.StringIO()
            with redirect_stdout(console):
                original = sys.stdout
                with tee_training_log(directory) as path:
                    print("第一次运行")
                self.assertIs(sys.stdout, original)
                with tee_training_log(directory):
                    print("第二次运行")
                self.assertIs(sys.stdout, original)
            self.assertIs(sys.stdout, before_stdout)
            self.assertIs(sys.stderr, before_stderr)
            self.assertEqual(path.read_text(encoding="utf-8"), "第一次运行\n第二次运行\n")

    def test_exception_traceback_is_logged_and_streams_are_restored(self):
        """未处理异常的文件回溯完整，异常仍传给调用方且不会替换其输出流。"""
        with tempfile.TemporaryDirectory() as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                before_stdout, before_stderr = sys.stdout, sys.stderr
                with self.assertRaisesRegex(RuntimeError, "评价失败"):
                    with tee_training_log(directory) as path:
                        print("异常之前")
                        raise RuntimeError("评价失败")
                self.assertIs(sys.stdout, before_stdout)
                self.assertIs(sys.stderr, before_stderr)
            content = path.read_text(encoding="utf-8")
            self.assertIn("异常之前", content)
            self.assertIn("Traceback (most recent call last)", content)
            self.assertIn("RuntimeError: 评价失败", content)
            # 交给普通 Python 异常入口显示回溯，避免上下文重复向终端打印。
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_path_validation_accepts_children_and_rejects_siblings(self):
        """名称前缀相同的兄弟目录不属于输出根目录，必须明确拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            validate_output_paths(root, root / "weights" / "experiment")
            with self.assertRaisesRegex(ValueError, "--output-dir"):
                validate_output_paths(root, Path(directory) / "outputs_other" / "weights")
            with self.assertRaisesRegex(ValueError, "--snapshots-folder"):
                validate_output_paths(root, root / ".." / "outside")
            self.assertFalse(root.exists())

    def test_path_validation_rejects_symlink_escape(self):
        """输出根目录内的符号链接不能把权重静默写到根目录之外。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            root.mkdir()
            outside = Path(directory) / "outside"
            outside.mkdir()
            (root / "escaped").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "output.root"):
                validate_output_paths(root, root / "escaped" / "weights")

    def test_main_records_initialization_fit_and_run_marker(self):
        """日志上下文覆盖模型初始化、评价器提示和实际训练入口。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            stdout, stderr = self.run_main(config, ["--max-steps", "12"])
            path = Path(config["output"]["root"]) / "logs" / "train.log"
            content = path.read_text(encoding="utf-8")
            for text in ("训练运行开始：", "日志文件：", "权重目录：", "模型初始化日志",
                         "评价器初始化警告", "训练调用 max_steps=12", "逐轮评价日志", "Training finished"):
                self.assertIn(text, content)
            self.assertIn("模型初始化日志", stdout)
            self.assertIn("评价器初始化警告", stderr)

    def test_output_dir_relocates_logs_and_default_weights(self):
        """覆盖输出根目录时权重路径自动跟随，不遗留原默认目录。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            root = Path(directory) / "custom" / "outputs"
            self.run_main(config, ["--output-dir", str(root)])
            received = FakeTrainer.received_configs[-1]
            self.assertEqual(received["output"]["root"], str(root.resolve()))
            self.assertEqual(received["train"]["snapshots_folder"], str(root.resolve() / "weights"))
            self.assertTrue((root / "logs" / "train.log").is_file())
            self.assertFalse(Path(config["output"]["root"]).exists())

    def test_output_dir_and_nested_snapshots_override_work_together(self):
        """两个选项可组合，保留用户指定的根目录内权重子目录。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            root = Path(directory) / "experiment" / "outputs"
            snapshots = root / "weights" / "run1"
            self.run_main(config, ["--snapshots-folder", str(snapshots), "--output-dir", str(root)])
            received = FakeTrainer.received_configs[-1]
            self.assertEqual(received["train"]["snapshots_folder"], str(snapshots.resolve()))
            self.assertTrue((root / "logs" / "train.log").is_file())

    def test_invalid_snapshots_override_does_not_create_log_or_trainer(self):
        """非法输出路径在启动模型前拒绝，不留下新的日志目录。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            before = len(FakeTrainer.received_configs)
            with self.assertRaisesRegex(ValueError, "--output-dir"):
                self.run_main(config, ["--snapshots-folder", str(Path(directory) / "outside")])
            self.assertEqual(len(FakeTrainer.received_configs), before)
            self.assertFalse(Path(config["output"]["root"]).exists())

    def test_invalid_training_override_does_not_create_log_or_trainer(self):
        """覆盖训练参数后再次校验，错误值不会进入初始化或日志上下文。"""
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            before = len(FakeTrainer.received_configs)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                self.run_main(config, ["--batch-size", "0"])
            self.assertEqual(len(FakeTrainer.received_configs), before)
            self.assertFalse(Path(config["output"]["root"]).exists())


if __name__ == "__main__":
    unittest.main()
