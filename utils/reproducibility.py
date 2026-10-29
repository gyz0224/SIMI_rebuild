"""统一随机种子及可安全序列化的随机状态。"""

import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic: bool) -> None:
    """在创建模型前设置随机源；None 仅用于显式的历史实验兼容。"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    # 不因关闭确定性而自动打开算法测速，避免额外改变旧数值策略。
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    """只保存张量和原生类型，兼容 torch.load 的 weights_only 安全读取。"""
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "keys": torch.from_numpy(keys.astype(np.int64)),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng_state(state: dict) -> None:
    """恢复 Python、NumPy、CPU，以及检查点中保存的 CUDA 随机状态。"""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["algorithm"],
        numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("完整恢复要求 CUDA 随机状态与当前可见设备数量一致")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def seed_worker(worker_id: int) -> None:
    """工作进程的基础种子由独立 loader generator 提供。"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
