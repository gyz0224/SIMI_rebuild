"""提供可替换的位平面提取接口，并保留原 SIMI 的二进制位平面约定。

提取接口接收带批次维的 RGB 张量，返回 24 个通道。
二进制实现完全在输入所在设备上进行整数张量运算，
不会使用 Python 逐个访问像素，也不会将图像复制到 CPU。

保留现有函数接口，以兼容数据集和模型的调用方式。
模式工厂仅内置 binary；其他模式仅保留接口，可显式注入后续实现。
提取器仍返回 Tensor[B, 24, H, W]，未来可靠性图应通过增强接口的
enhancement_guidance 或 extra_guidance 独立传递。
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


BIT_MODES = ("binary", "gray", "learnable", "reliable")


def validate_bit_mode(mode: str) -> str:
    """校验位平面模式，拒绝非字符串或尚未约定的模式名称。"""
    if not isinstance(mode, str) or mode not in BIT_MODES:
        raise ValueError(f"bit_mode 必须是以下模式之一：{', '.join(BIT_MODES)}")
    return mode


def _check_rgb_shape(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim not in (3, 4) or image.shape[-3] != 3:
        raise ValueError("image must have shape [3, H, W] or [B, 3, H, W]")


class BitPlaneExtractor(nn.Module, ABC):
    """扩展接口：将带批次维的 RGB 图像转换为 24 通道位平面特征。

    子类实现 ``forward(image)``，即可作为普通 PyTorch 模块传递和调用。
    必须保留 RGB 通道顺序、批次大小和空间尺寸。
    输出与输入位于同一设备，形状为 [B, 24, H, W]。

    后续实现可以继承此接口，提供格雷码特征、可学习特征或考虑可靠性的特征。
    此处暂不实现这些方案。当前二进制提取器不保存状态；
    后续的可学习模块可以注册自身参数，并参与自动求导。
    """

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """从形状为 [B, 3, H, W] 的张量中提取 24 个特征平面。"""
        raise NotImplementedError


class BinaryBitPlaneExtractor(BitPlaneExtractor):
    """提取与原 SIMI 数据加载器完全一致的二进制位平面。

    输入：
        形状为 [B, 3, H, W]，通道顺序为 RGB。
        优先使用取值为 [0, 255] 的 uint8 像素，以精确分解原始图像。
        也接受取值为 [0, 1] 的归一化浮点 RGB，
        并沿用现有的 round(image * 255) 转换约定。
        浮点输入必须全部为有限值，且位于上述范围内；
        不会裁剪数值，也不会根据图像内容重新缩放。
        特别地，取值为 [0, 255] 的浮点图像不属于归一化浮点输入格式。

    输出：
        形状为 [B, 24, H, W]，类型为 float32，与输入位于同一设备，
        数值严格为 0 或 1。
        通道顺序为 R7, R6, ..., R0, G7, ..., G0, B7, ..., B0。

    不包含参数、缓冲区、随机操作、平滑操作或可学习层。
    这种离散的 uint8 分解不会向输入反向传播梯度。
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _check_rgb_shape(image)
        if image.ndim != 4:
            raise ValueError("BinaryBitPlaneExtractor requires shape [B, 3, H, W]")

        # 原 SIMI 将每个 uint8 像素格式化为八位二进制字符串。
        # 因此，数据集应在 RGB 归一化之前，直接传入缩放后的 uint8 图像。
        # 浮点分支保留重构项目现有的归一化 RGB 便捷接口。
        if image.dtype == torch.uint8:
            pixels = image
        elif torch.is_floating_point(image):
            pixels = torch.round(image * 255).to(torch.uint8)
        else:
            raise TypeError("image must be uint8 RGB or normalized floating-point RGB")

        # 第 7 位是最高有效位。在 pixels.device 上创建移位量，
        # 可让 CUDA 输入始终留在 GPU，避免复制到 CPU 或通过 .item() 触发同步。
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=pixels.device)

        # [B, 3, H, W] -> [B, 3, 1, H, W]。通过广播八个移位量，
        # 使用一个张量表达式提取所有通道、所有像素的全部位：
        # 计算关系：plane[b, c, k, h, w] = (pixel[b, c, h, w] >> (7 - k)) & 1
        # 整数移位与掩码运算和原二进制字符串转换完全等价，
        # 包括对较小像素值补齐前导零的行为。
        planes = (pixels.unsqueeze(2) >> shifts.view(1, 1, 8, 1, 1)) & 1

        # 按通道维、位维的顺序合并这两个维度，让每种颜色的八个位保持相邻。
        # 仅在完成整数提取后转换为 float32；0 和 1 均可精确表示，
        # 与原数据加载器调用 float() 后的结果一致。
        return planes.flatten(start_dim=1, end_dim=2).to(torch.float32)


def build_bitplane_extractor(bit_mode: str = "binary", *, extractor=None) -> BitPlaneExtractor:
    """构造二进制提取器，或使用调用者显式注入的接口实现。

    当前仅实现 binary；gray、learnable、reliable 仅为后续扩展接口，
    未注入实现时明确报错，不自动退回二进制算法。
    注入对象必须继承 BitPlaneExtractor，并返回 Tensor[B, 24, H, W]。
    未来的可靠性图应通过 enhancement_guidance 或 extra_guidance
    独立传给增强网络，不改变提取器的张量返回接口。
    """
    validate_bit_mode(bit_mode)
    if extractor is not None:
        if not isinstance(extractor, BitPlaneExtractor):
            raise TypeError("extractor 必须是 BitPlaneExtractor 的实例")
        return extractor
    if bit_mode == "binary":
        return BinaryBitPlaneExtractor()
    raise NotImplementedError(f"位平面模式 {bit_mode!r} 尚未实现，请显式注入 BitPlaneExtractor 实例")


def decompose_bitplanes(image: torch.Tensor) -> torch.Tensor:
    """兼容 CHW 或 BCHW 格式 uint8 RGB 图像的包装函数。

    类接口始终要求输入包含批次维。此包装函数临时添加大小为 1 的批次维，
    并在提取后移除该维度，从而保留数据集原有的无批次输入输出约定。
    """
    _check_rgb_shape(image)
    if image.dtype != torch.uint8:
        raise TypeError("decompose_bitplanes requires a uint8 tensor")
    unbatched = image.ndim == 3
    batched = image.unsqueeze(0) if unbatched else image
    planes = BinaryBitPlaneExtractor()(batched)
    return planes.squeeze(0) if unbatched else planes


def bitplanes_from_rgb(image: torch.Tensor) -> torch.Tensor:
    """将归一化浮点 RGB 乘以 255 并通过 round 舍入为 uint8，再进行位平面分解。

    调用者需提供取值为 [0, 1] 的有限 RGB 数值。
    数据加载时直接使用原始 uint8 图像，避免不必要的浮点往返转换。
    """
    _check_rgb_shape(image)
    if not torch.is_floating_point(image):
        raise TypeError("bitplanes_from_rgb requires a floating-point tensor")
    unbatched = image.ndim == 3
    batched = image.unsqueeze(0) if unbatched else image
    planes = BinaryBitPlaneExtractor()(batched)
    return planes.squeeze(0) if unbatched else planes
