"""将训练标准输出和错误输出同时写入终端与 outputs 日志文件。"""

from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import traceback


class TeeStream:
    """保留原输出流行为，并即时追加一份文本到共享日志。

    两条输出流共用同一个锁，避免标准输出与警告在日志中交错写入。
    编码、终端检测等属性仍交给原始流，兼容 print、warnings 和进度显示。
    """

    def __init__(self, original, log_stream, lock):
        self.original = original
        self.log_stream = log_stream
        self.lock = lock

    def write(self, text):
        """每次输出都刷新文件，方便训练期间直接查看最新日志。"""
        with self.lock:
            self.log_stream.write(text)
            self.log_stream.flush()
            self.original.write(text)
        return len(text)

    def flush(self):
        """同时刷新终端与日志，不关闭由调用方拥有的原始流。"""
        with self.lock:
            self.log_stream.flush()
            self.original.flush()

    def __getattr__(self, name):
        """转发 encoding、isatty 等属性，保留原终端的运行语义。"""
        return getattr(self.original, name)


def validate_output_paths(output_root, snapshots_folder) -> None:
    """拒绝将权重写到输出根目录之外；在创建日志或启动训练前调用。"""
    root = Path(output_root).expanduser().resolve()
    snapshots = Path(snapshots_folder).expanduser().resolve()
    try:
        snapshots.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"权重目录必须位于 output.root 内：{snapshots}；当前 output.root={root}。"
            "请使用 --output-dir 指定新的输出根目录，或将 --snapshots-folder 设为其子目录。"
        ) from error


@contextmanager
def tee_training_log(output_root):
    """追加记录训练与评价日志，并在任何退出路径恢复 stdout/stderr。

    日志固定为 output_root/logs/train.log；断点续训与重复运行均不会截断
    已有日志。Trainer 初始化、LPIPS 警告和训练过程都应放在此上下文内。
    未处理异常的回溯直接记入文件，随后重新抛出，由 Python 正常输出到
    终端，避免终端出现两份相同回溯。
    """
    log_path = Path(output_root).expanduser().resolve() / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    lock = threading.RLock()
    with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        stdout = TeeStream(original_stdout, log_stream, lock)
        stderr = TeeStream(original_stderr, log_stream, lock)
        sys.stdout, sys.stderr = stdout, stderr
        try:
            yield log_path
        except BaseException as error:
            # 外层 Python 的异常处理发生在上下文退出之后，需提前保留文件回溯。
            if not isinstance(error, SystemExit) or error.code not in (None, 0):
                traceback.print_exception(type(error), error, error.__traceback__, file=log_stream)
                log_stream.flush()
            raise
        finally:
            # 先恢复流，再刷新缓存；即使刷新失败也不会污染后续训练或测试输出。
            sys.stdout, sys.stderr = original_stdout, original_stderr
            stdout.flush()
            stderr.flush()
