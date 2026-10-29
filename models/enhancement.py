"""七阶段 DEA 曲线增强模块，以及用于历史权重复现的旧卷积结构。"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .bit_encoder import BitEncoder
from .blocks import DEABlock, initialize_like_original


ENHANCEMENT_GUIDANCE_MODES = ("none", "snr", "reliability")
ENHANCEMENT_BACKBONES = ("dea", "legacy_cnn")


def validate_enhancement_backbone(enhancement_backbone: str) -> str:
    """校验增强主干；旧卷积结构只用于历史结果和检查点复现。"""
    if (not isinstance(enhancement_backbone, str)
            or enhancement_backbone not in ENHANCEMENT_BACKBONES):
        raise ValueError("enhancement_backbone 必须是 dea 或 legacy_cnn")
    return enhancement_backbone


def validate_enhancement_guidance(enhancement_guidance: str) -> str:
    """校验引导模式名称；合法候选不表示相应研究模块已经实现。"""
    if (not isinstance(enhancement_guidance, str)
            or enhancement_guidance not in ENHANCEMENT_GUIDANCE_MODES):
        raise ValueError("enhancement_guidance 必须是 none、snr 或 reliability")
    return enhancement_guidance


class EnhancementCurveNetwork(nn.Module):
    """底层曲线预测与递推核心，默认使用七个串联 DEA 块。

    特征输入为已经完成原始平滑、注意力和融合处理的 27 通道张量。
    每一级预测 3 通道 A 和 B，共七级，拼接后各得到 21 通道参数图。
    A 保留 tanh 激活；B 保留 sigmoid 激活及原始的 0.5 缩放、0.5 偏移。
    DEA 块保持 32 通道，先将融合后的 27 通道映射到 32 通道。
    legacy_cnn 显式保留原源码结构与首级 B 共享行为，供旧实验复现。
    """

    def __init__(self, enhancement_backbone: str = "dea", *,
                 activation_checkpointing: bool = False):
        super().__init__()
        self.enhancement_backbone = validate_enhancement_backbone(enhancement_backbone)
        if not isinstance(activation_checkpointing, bool):
            raise ValueError("activation_checkpointing 必须是布尔值")
        self.activation_checkpointing = activation_checkpointing
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        if self.enhancement_backbone == "dea":
            # DEA 块包含残差相加，输入与输出必须同为 32 通道。
            # 通道投影不计入 D；随后仍执行完整的七个 DEA 块。
            self.input_projection = nn.Conv2d(27, 32, 3, 1, 1, bias=True)
        for stage in range(1, 8):
            if self.enhancement_backbone == "dea":
                block = DEABlock(32)
            else:
                in_channels = 27 if stage == 1 else 32
                block = nn.Conv2d(in_channels, 32, 3, 1, 1, bias=True)
            setattr(self, f"e_conv{stage}", block)
            setattr(self, f"e_conv{stage}_a", nn.Conv2d(32, 3, 3, 1, 1, bias=True))
            setattr(self, f"e_conv{stage}_b", nn.Conv2d(32, 3, 3, 1, 1, bias=True))

    def predict_curves(self, fused: torch.Tensor):
        """依次预测七组参数，返回 A、B 和末级 32 通道特征。

        保留先生成全部曲线参数、再对 RGB 递推更新的原始执行顺序，
        不将中间增强图重新输入特征主干。默认每一级 A、B 分别预测。
        """
        features = (self.input_projection(fused)
                    if self.enhancement_backbone == "dea" else fused)
        curves_a = []
        curves_b = []
        for stage in range(1, 8):
            block = getattr(self, f"e_conv{stage}")
            if (self.enhancement_backbone == "dea" and self.activation_checkpointing
                    and self.training and torch.is_grad_enabled()):
                # 反向时重新计算块内部特征，保留相同运算、梯度和参数。
                # 非重入实现支持常规 backward；推理及冻结分支不使用检查点。
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
            if self.enhancement_backbone == "legacy_cnn":
                features = self.relu(features)
            head_a = getattr(self, f"e_conv{stage}_a")
            head_b = getattr(self, f"e_conv{stage}_b")
            if self.enhancement_backbone == "legacy_cnn" and stage == 1:
                # 只有显式历史模式复现旧源码的首级共享，不影响默认修正路径。
                head_b = head_a
            curves_a.append(self.tanh(head_a(features)))
            curves_b.append(self.sigmoid(head_b(features)) * 0.5 + 0.5)
        curve_a = torch.cat(curves_a, dim=1)
        curve_b = torch.cat(curves_b, dim=1)
        return curve_a, curve_b, features

    @staticmethod
    def apply_curves(image: torch.Tensor, curve_a: torch.Tensor,
                     curve_b: torch.Tensor) -> torch.Tensor:
        """执行原始的七次递推更新，保持每次计算的浮点运算顺序。

        对第 i 级的三个 RGB 通道，使用对应 A_i 和 B_i 更新当前图像：
        I_next = I + A_i / (1 + exp(-10 * (-I + B_i - 0.1)))
                   * I * (B_i - I) * (1 / B_i)。
        这里直接保留代码中的 Curve 公式，不进行代数化简或额外裁剪。
        AMP 只降低曲线预测网络的计算精度；递推含指数、倒数以及连续七次
        更新，因此在 AMP 开启时将图像和曲线转换为 FP32 后计算。
        未开启 AMP 时保留输入原有类型，维持历史 FP32/FP64 输出及梯度。
        """
        use_amp = torch.is_autocast_enabled(image.device.type)
        if use_amp:
            # 转换仍属于计算图，梯度会正常回传至原始的半精度曲线预测头。
            # 只替换局部张量，不改变 forward 返回的 A、B 或末级特征类型。
            image = image.float()
            curve_a = curve_a.float()
            curve_b = curve_b.float()
        with torch.autocast(device_type=image.device.type, enabled=False):
            enhanced = image
            for stage in range(7):
                # 保留原始运算顺序，避免代数化简改变浮点计算结果。
                enhanced = enhanced + curve_a[:, 3 * stage:3 * stage + 3, :, :] * 1 / (
                    1 + torch.exp(-10 * (-enhanced + curve_b[:, 3 * stage:3 * stage + 3, :, :] - 0.1))
                ) * enhanced * (
                    curve_b[:, 3 * stage:3 * stage + 3, :, :] - enhanced
                ) * (1 / curve_b[:, 3 * stage:3 * stage + 3, :, :])
        return enhanced

    def forward(self, rgb: torch.Tensor, fused: torch.Tensor):
        """保留底层旧接口：返回增强图、两组参数图和末级特征。"""
        curve_a, curve_b, features = self.predict_curves(fused)
        enhanced = self.apply_curves(rgb, curve_a, curve_b)
        return enhanced, curve_a, curve_b, features


class CurveEnhancementNetwork(nn.Module):
    """RGB 与位平面输入的独立曲线增强网络。

    输入：
        image：归一化 RGB 图像，形状为 [B, 3, H, W]。
        bit_feature：形状为 [B, 24, H, W] 的位平面输入，与 RGB 的批次、
            空间尺寸、设备和浮点类型一致。默认使用 BinaryBitPlaneExtractor
            产生的原始位平面；本网络内部仍会执行原有的平滑和融合过程，
            因此这里应传入尚未经过本网络平滑、注意力与融合的位平面张量。
        extra_guidance：未来的可靠性图、SNR 图或自适应曲线控制扩展入口。
            默认为 None；当前未实现引导算法，非空输入会明确报错。

    输出：
        forward 返回 [B, 3, H, W] 的增强图像。
        forward_with_aux 同时返回增强图、A 参数图、B 参数图和末级特征，
        供现有无监督损失和去噪分支使用。

    默认组合 BitEncoder 与 DEA 曲线核心，保持七级预测头及原始递推公式。
    显式 legacy_cnn 保留旧初始化顺序与权重结构，不能与 DEA 权重混用。
    """

    def __init__(self, enhancement_backbone: str = "dea", *,
                 activation_checkpointing: bool = False):
        super().__init__()
        self.enhancement_backbone = validate_enhancement_backbone(enhancement_backbone)
        # 先创建曲线预测模块，再创建编码模块，历史模式保持原初始化顺序。
        self.curve_network = EnhancementCurveNetwork(
            self.enhancement_backbone, activation_checkpointing=activation_checkpointing,
        )
        self.bit_encoder = BitEncoder()
        if self.enhancement_backbone == "legacy_cnn":
            initialize_like_original(self)
        else:
            # DEA 块沿用官方构造初始化，不把旧单层卷积初始化强加到五分支卷积。
            # 原编码器、曲线头和输入投影仍使用原初始化函数，避免多级尺度膨胀。
            initialize_like_original(self.bit_encoder)
            initialize_like_original(self.curve_network.input_projection)
            for stage in range(1, 8):
                initialize_like_original(getattr(self.curve_network, f"e_conv{stage}_a"))
                initialize_like_original(getattr(self.curve_network, f"e_conv{stage}_b"))

    @staticmethod
    def _check_inputs(image: torch.Tensor, bit_feature: torch.Tensor) -> None:
        """校验接口元信息，不读取像素数值或触发 GPU 同步。"""
        if not isinstance(image, torch.Tensor) or not isinstance(bit_feature, torch.Tensor):
            raise TypeError("image 和 bit_feature 必须是 torch.Tensor")
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image 的形状必须是 [B, 3, H, W]")
        if bit_feature.ndim != 4 or bit_feature.shape[1] != 24:
            raise ValueError("bit_feature 的形状必须是 [B, 24, H, W]")
        if image.shape[0] != bit_feature.shape[0] or image.shape[2:] != bit_feature.shape[2:]:
            raise ValueError("RGB 与位平面的批次数和空间尺寸必须一致")
        if image.device != bit_feature.device or image.dtype != bit_feature.dtype:
            raise ValueError("RGB 与位平面的设备和数据类型必须一致")
        if not torch.is_floating_point(image):
            raise TypeError("RGB 与位平面必须是浮点张量")

    def _prepare_features(self, image: torch.Tensor, bit_feature: torch.Tensor,
                          extra_guidance=None) -> torch.Tensor:
        """生成原有的 27 通道融合特征，作为后续研究的可覆盖入口。

        未来子类可以覆盖本方法，将可靠性或 SNR 信息用于特征准备；
        自适应曲线控制也可以通过覆盖 forward_with_aux 扩展预测与递推。
        当前只实现原始路径，不静默忽略尚未实现的引导输入。
        """
        if extra_guidance is not None:
            raise NotImplementedError("extra_guidance 当前仅为预留接口，尚未实现引导算法")
        return self.bit_encoder(image, bit_feature)

    def forward_with_aux(self, image: torch.Tensor, bit_feature: torch.Tensor,
                         extra_guidance=None):
        """执行所选增强主干，返回训练和去噪所需的四项结果。"""
        self._check_inputs(image, bit_feature)
        if self.enhancement_backbone == "dea" and min(image.shape[2:]) <= 3:
            raise ValueError("DEA 的反射填充要求图像高和宽均大于 3")
        fused = self._prepare_features(image, bit_feature, extra_guidance)
        return self.curve_network(image, fused)

    def forward(self, image: torch.Tensor, bit_feature: torch.Tensor,
                extra_guidance=None) -> torch.Tensor:
        """根据 RGB 和 24 通道位平面生成增强图像，默认执行 DEA 路径。"""
        return self.forward_with_aux(image, bit_feature, extra_guidance)[0]

    def extract_features(self, rgb: torch.Tensor,
                         bitplanes: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.bit_encoder.extract_features(rgb, bitplanes)


class EnhancementNetwork(CurveEnhancementNetwork):
    """现有 SIMI 调用链的兼容接口，保留原来的四项输出。

    构造、参数和计算流程全部复用 CurveEnhancementNetwork；这里仅适配
    返回值，保持现有 SIMI、损失函数与去噪网络所需的调用接口。
    """

    def forward(self, rgb: torch.Tensor, bitplanes: torch.Tensor,
                extra_guidance=None):
        """返回增强 RGB、A、B 和末级特征，与原始 light_net 一致。"""
        return self.forward_with_aux(rgb, bitplanes, extra_guidance)


def build_enhancement_network(enhancement_guidance: str = "none", *, network=None,
                              enhancement_backbone: str = "dea",
                              activation_checkpointing: bool = False):
    """创建增强分支，或接入研究者提供的兼容增强网络。

    默认创建 DEA 无引导网络。snr 和 reliability 仅保留模式名称，
    未提供具体实现时明确报错，不会将其静默退回无引导路径。
    未来可以注入继承 EnhancementNetwork 的实现，覆盖 _prepare_features
    或 forward_with_aux 使用 extra_guidance，同时保持四项辅助输出。
    独立 CurveEnhancementNetwork 的 forward 返回单个 RGB 张量，
    集成 SIMI 时应使用其 EnhancementNetwork 兼容子类。
    """
    validate_enhancement_guidance(enhancement_guidance)
    validate_enhancement_backbone(enhancement_backbone)
    if not isinstance(activation_checkpointing, bool):
        raise ValueError("activation_checkpointing 必须是布尔值")
    if network is not None:
        if not isinstance(network, EnhancementNetwork):
            raise TypeError("注入的增强网络必须继承 EnhancementNetwork 并保持四项输出接口")
        if network.enhancement_backbone != enhancement_backbone:
            raise ValueError("注入增强网络的 enhancement_backbone 与配置不一致")
        network.curve_network.activation_checkpointing = activation_checkpointing
        return network
    if enhancement_guidance != "none":
        raise NotImplementedError(f"增强引导模式 {enhancement_guidance!r} 仅为预留接口，尚未实现")
    return EnhancementNetwork(enhancement_backbone, activation_checkpointing=activation_checkpointing)
