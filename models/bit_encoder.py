"""位平面平滑与 RGB 特征融合。"""

import torch
from torch import nn

from .attention import CBAM
from .blocks import ConvSiLU


class BitEncoder(nn.Module):
    """保留原始实现的两个平滑模块（S=2）与三个融合卷积层。"""

    def __init__(self):
        super().__init__()
        # 按原始顺序注册模块，同时保持旧版优化器的参数顺序。
        self.smoothing = nn.ModuleList([
            ConvSiLU(24, 24),
            ConvSiLU(24, 24, dilation=2),
        ])
        self.fusion = nn.ModuleList([
            ConvSiLU(27, 27),
            ConvSiLU(27, 27, dilation=2),
            ConvSiLU(27, 27, dilation=3),
        ])
        self.attention = CBAM(27)

    def extract_features(self, rgb: torch.Tensor,
                         bitplanes: torch.Tensor) -> dict[str, torch.Tensor]:
        smoothed = bitplanes
        for layer in self.smoothing:
            smoothed = layer(smoothed)
        fused = torch.cat((rgb, smoothed), dim=1)
        fused = self.attention(fused)
        for layer in self.fusion:
            fused = layer(fused)
        return {
            "rgb": rgb,
            "raw_bitplanes": bitplanes,
            "bit_plane_features": smoothed,
            "fused_features": fused,
        }

    def forward(self, rgb: torch.Tensor, bitplanes: torch.Tensor) -> torch.Tensor:
        return self.extract_features(rgb, bitplanes)["fused_features"]
