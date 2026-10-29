"""测量一次模型前向；CUDA 在计时边界同步，图像读写与指标不计入。"""

import time
from typing import Callable

import torch


def measure_forward(forward: Callable, device):
    """执行并返回前向输出和耗时秒数，只同步所选 CUDA 设备。"""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - started


def benchmark_forward(forward: Callable, device, *, warmup: int = 0):
    """先执行不计时的预热，再测量同一阶段的一次前向。

    预热输出被丢弃，只有测量所得输出参与保存和质量指标计算。
    调用者负责设置 eval 和 inference_mode，保持模型执行策略明确。
    """
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup 必须是非负整数")
    for _ in range(warmup):
        forward()
    return measure_forward(forward, device)
