"""原始 SIMI 实现使用的空间与通道注意力模块。"""

import torch
from torch import nn


class CBAM(nn.Module):
    """依次应用通道注意力和空间注意力，保持原始数学运算不变。"""

    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel,
            padding=spatial_kernel // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.is_cuda and torch.are_deterministic_algorithms_enabled():
            # 有限特征的全局最大值及首个最大值位置与原池化完全相同。
            # 原 CUDA 池化反向不支持严格确定性；仅接受有限数值，保证并列最大值梯度一致。
            if not bool(torch.isfinite(features).all()):
                raise ValueError("严格 CUDA 通道注意力只接受有限数值的特征")
            pooled_maximum = features.flatten(2).max(dim=2).values.reshape(
                features.shape[0], features.shape[1], 1, 1,
            )
        else:
            pooled_maximum = self.max_pool(features)
        maximum = self.mlp(pooled_maximum)
        average = self.mlp(self.avg_pool(features))
        channel_attention = self.sigmoid(maximum + average)
        features = channel_attention * features
        maximum, _ = torch.max(features, dim=1, keepdim=True)
        average = torch.mean(features, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.conv(torch.cat([maximum, average], dim=1)))
        return spatial_attention * features


class DEAChannelAttention(nn.Module):
    """DEA-Net 的通道注意力，输出未经 sigmoid 的通道响应。"""

    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        if reduction < 1 or dim // reduction < 1:
            raise ValueError("DEA 通道注意力的 reduction 必须为正，且不能大于通道数")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.ca(self.gap(features))


class DEASpatialAttention(nn.Module):
    """DEA-Net 的空间注意力，对通道均值和最大值进行反射填充卷积。"""

    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average = torch.mean(features, dim=1, keepdim=True)
        maximum, _ = torch.max(features, dim=1, keepdim=True)
        return self.sa(torch.cat([average, maximum], dim=1))


class DEAPixelAttention(nn.Module):
    """将每个通道及其注意力响应交错配对，再计算内容引导的像素权重。"""

    def __init__(self, dim: int):
        super().__init__()
        self.pa2 = nn.Conv2d(
            2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        # [B, C, 2, H, W] 合并通道维和配对维，保留 [x0, a0, x1, a1, ...]。
        # 分组卷积每组只读取同一通道的特征及注意力，不能按两个完整张量拼接。
        paired = torch.cat([features.unsqueeze(2), attention.unsqueeze(2)], dim=2).flatten(1, 2)
        return self.sigmoid(self.pa2(paired))


__all__ = ["CBAM", "DEAChannelAttention", "DEASpatialAttention", "DEAPixelAttention"]
