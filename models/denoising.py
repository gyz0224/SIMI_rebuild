"""原始实现的默认去噪流程，输入特征与增强网络的梯度断开。"""

import torch
from torch import nn

from .attention import CBAM
from .blocks import ConvSiLU, initialize_like_original


class DenoisingNetwork(nn.Module):
    """保留原始实现的 35 通道特征细化模块与七层 RGB 解码网络。"""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # 保留未使用的批归一化参数和缓冲区，以兼容旧版模型及优化器状态。
        self.batch_norm = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(35, 32, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        self.attention = CBAM(35)
        self.refinement = nn.ModuleList([
            ConvSiLU(35, 35),
            ConvSiLU(35, 35, dilation=2),
            ConvSiLU(35, 35, dilation=3),
        ])
        initialize_like_original(self)

    def forward(self, enhanced: torch.Tensor,
                enhancement_features: torch.Tensor) -> torch.Tensor:
        features = torch.cat((enhanced, enhancement_features), dim=1)
        features = self.attention(features)
        for layer in self.refinement:
            features = layer(features)
        stage1 = self.relu(self.conv1(features))
        stage2 = self.relu(self.conv2(stage1))
        stage3 = self.relu(self.conv3(stage2))
        stage4 = self.relu(self.conv4(stage3))
        stage5 = self.relu(self.conv5(torch.cat([stage3, stage4], 1)))
        stage6 = self.relu(self.conv6(torch.cat([stage2, stage5], 1)))
        return self.sigmoid(self.conv7(torch.cat([stage1, stage6], 1)))
