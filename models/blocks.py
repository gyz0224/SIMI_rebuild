"""卷积模块与权重初始化，保持原始代码的行为。"""

import torch
from torch import nn
from torch.nn import functional as F

from .attention import DEAChannelAttention, DEAPixelAttention, DEASpatialAttention


class ConvSiLU(nn.Module):
    """原始 ConvBNReLU 模块实际只包含卷积和 SiLU 激活函数。"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.silu = nn.SiLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.silu(self.conv(features))


def initialize_like_original(module: nn.Module) -> None:
    """保持原始模块遍历顺序和随机数生成顺序，保留卷积偏置的初始值。"""
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(child.weight)
        elif isinstance(child, nn.BatchNorm2d):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)


class _CentralDifferenceConv(nn.Module):
    """中心差分分支，将卷积核总和从中心权重中减去。"""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def get_weight(self):
        weight = self.conv.weight
        flattened = weight.flatten(2)
        transformed = weight.new_zeros(flattened.shape)
        transformed[:, :, :] = flattened[:, :, :]
        transformed[:, :, 4] = flattened[:, :, 4] - flattened[:, :, :].sum(2)
        return transformed.reshape_as(weight), self.conv.bias


class _HorizontalDifferenceConv(nn.Module):
    """水平差分分支，以一维卷积参数构造左右符号相反的二维核。"""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, bias=True)

    def get_weight(self):
        weight = self.conv.weight
        transformed = weight.new_zeros(weight.shape[0], weight.shape[1], 9)
        transformed[:, :, [0, 3, 6]] = weight
        transformed[:, :, [2, 5, 8]] = -weight
        return transformed.reshape(weight.shape[0], weight.shape[1], 3, 3), self.conv.bias


class _VerticalDifferenceConv(nn.Module):
    """垂直差分分支，以一维卷积参数构造上下符号相反的二维核。"""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, bias=True)

    def get_weight(self):
        weight = self.conv.weight
        transformed = weight.new_zeros(weight.shape[0], weight.shape[1], 9)
        transformed[:, :, [0, 1, 2]] = weight
        transformed[:, :, [6, 7, 8]] = -weight
        return transformed.reshape(weight.shape[0], weight.shape[1], 3, 3), self.conv.bias


class _AngularDifferenceConv(nn.Module):
    """角向差分分支，按官方角向置换顺序相减。"""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def get_weight(self):
        weight = self.conv.weight
        flattened = weight.flatten(2)
        transformed = flattened - flattened[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        return transformed.reshape_as(weight), self.conv.bias


class DEConv(nn.Module):
    """DEA-Net 的五分支细节增强卷积，不包含未使用的径向差分分支。

    中心、水平、垂直、角向差分及普通卷积核相加后执行一次二维卷积。
    转换张量继承权重的设备和数值类型，支持 CPU、CUDA 及模型类型转换。
    子模块命名与官方训练实现一致，便于严格对照权重及梯度。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv1_1 = _CentralDifferenceConv(dim)
        self.conv1_2 = _HorizontalDifferenceConv(dim)
        self.conv1_3 = _VerticalDifferenceConv(dim)
        self.conv1_4 = _AngularDifferenceConv(dim)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weight1, bias1 = self.conv1_1.get_weight()
        weight2, bias2 = self.conv1_2.get_weight()
        weight3, bias3 = self.conv1_3.get_weight()
        weight4, bias4 = self.conv1_4.get_weight()
        weight5, bias5 = self.conv1_5.weight, self.conv1_5.bias
        weight = weight1 + weight2 + weight3 + weight4 + weight5
        bias = bias1 + bias2 + bias3 + bias4 + bias5
        return F.conv2d(features, weight, bias, stride=1, padding=1, groups=1)


class DEABlock(nn.Module):
    """官方训练式 DEA 块：细节增强卷积、双残差和内容引导注意力。

    顺序为 DEConv→ReLU→残差→普通卷积→CGA→残差。
    空间和像素注意力使用反射填充，因此输入的高和宽必须大于 3。
    保留 PyTorch 卷积的默认构造初始化，不调用或改变旧初始化函数。
    """

    def __init__(self, dim: int, kernel_size: int = 3, reduction: int = 8):
        super().__init__()
        self.conv1 = DEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, bias=True)
        self.sa = DEASpatialAttention()
        self.ca = DEAChannelAttention(dim, reduction)
        self.pa = DEAPixelAttention(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.act1(self.conv1(features))
        residual = residual + features
        residual = self.conv2(residual)
        channel_attention = self.ca(residual)
        spatial_attention = self.sa(residual)
        initial_attention = spatial_attention + channel_attention
        pixel_attention = self.pa(residual, initial_attention)
        residual = residual * pixel_attention
        return residual + features


__all__ = ["ConvSiLU", "initialize_like_original", "DEConv", "DEABlock"]
