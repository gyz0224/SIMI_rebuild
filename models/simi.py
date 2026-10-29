"""组合 SIMI 模型各模块，严格区分 DEA 与历史卷积结构的检查点。"""

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn

from datasets.bitplane import build_bitplane_extractor, validate_bit_mode

from .denoising import DenoisingNetwork
from .enhancement import (
    build_enhancement_network, validate_enhancement_backbone,
    validate_enhancement_guidance,
)


def _legacy_key(current_key: str) -> str:
    prefix_pairs = (
        ("light_net.curve_network.", "light_net."),
        ("light_net.bit_encoder.smoothing.", "light_net.Conv."),
        ("light_net.bit_encoder.fusion.", "light_net.Conv1."),
        ("light_net.bit_encoder.attention.", "light_net.CBAMLayer."),
        ("denoise.batch_norm.", "denoise.bn."),
        ("denoise.attention.", "denoise.CBAMLayer2."),
        ("denoise.refinement.", "denoise.Conv2."),
    )
    for current_prefix, legacy_prefix in prefix_pairs:
        if current_key.startswith(current_prefix):
            return legacy_prefix + current_key[len(current_prefix):]
    for stage in range(1, 8):
        current_prefix = f"denoise.conv{stage}."
        if current_key.startswith(current_prefix):
            return f"denoise.d_e_conv{stage}." + current_key[len(current_prefix):]
    return current_key


def _check_state_dict(state_dict: Mapping, expected: Mapping) -> None:
    supplied = set(state_dict)
    required = set(expected)
    missing = sorted(required - supplied)
    unexpected = sorted(supplied - required)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint keys do not match SIMI: missing={missing}, unexpected={unexpected}"
        )
    for key, expected_tensor in expected.items():
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"Checkpoint value {key!r} must be a tensor")
        if tensor.shape != expected_tensor.shape:
            raise RuntimeError(
                f"Checkpoint shape mismatch for {key}: "
                f"received {tuple(tensor.shape)}, expected {tuple(expected_tensor.shape)}"
            )


class SIMI(nn.Module):
    """保持原始兼容性的 RGB 与位平面增强、去噪模型。

    forward 返回 (enhanced_rgb, curve_a, curve_b, denoised_rgb)，
    与 model_new.bitenhanced 一致。enhance 返回最终去噪后的 RGB 图像。
    默认 binary + none + dea 使用七级 DEA 和独立 A/B 头；
    legacy_cnn 显式复现历史卷积网络，其余研究模式仅保留工厂与注入接口。
    extra_guidance 可将未来的可靠性图或 SNR 图传给增强分支，
    本类不生成引导图，也不实现引导算法。
    """

    def __init__(self, bit_mode: str = "binary", enhancement_guidance: str = "none",
                 *, enhancement_backbone: str = "dea",
                 activation_checkpointing: bool = False,
                 bitplane_extractor=None, enhancement_network=None):
        super().__init__()
        self.bit_mode = validate_bit_mode(bit_mode)
        self.enhancement_guidance = validate_enhancement_guidance(enhancement_guidance)
        self.enhancement_backbone = validate_enhancement_backbone(enhancement_backbone)
        extractor = build_bitplane_extractor(bit_mode, extractor=bitplane_extractor)
        self.light_net = build_enhancement_network(
            enhancement_guidance, network=enhancement_network,
            enhancement_backbone=self.enhancement_backbone,
            activation_checkpointing=activation_checkpointing,
        )
        self.denoise = DenoisingNetwork()
        # 默认提取器没有参数或随机操作，最后注册以保留原网络参数顺序。
        # 该模块只有模型一个所有者，训练器与推理共用此实例。
        self.bitplane_extractor = extractor

    def _prepare_bitplanes(self, rgb: torch.Tensor,
                           bitplanes: torch.Tensor | None) -> torch.Tensor:
        if bitplanes is None:
            bitplanes = self.bitplane_extractor(rgb)
        return bitplanes

    def forward(self, rgb: torch.Tensor, bitplanes: torch.Tensor | None = None,
                extra_guidance=None):
        bitplanes = self._prepare_bitplanes(rgb, bitplanes)
        enhanced, curve_a, curve_b, features = self.light_net(rgb, bitplanes, extra_guidance=extra_guidance)
        denoised = self.denoise(enhanced.detach(), features.detach())
        return enhanced, curve_a, curve_b, denoised

    def enhance(self, rgb: torch.Tensor,
                bitplanes: torch.Tensor | None = None, extra_guidance=None) -> torch.Tensor:
        return self(rgb, bitplanes, extra_guidance=extra_guidance)[3]

    def extract_features(self, rgb: torch.Tensor,
                         bitplanes: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bitplanes = self._prepare_bitplanes(rgb, bitplanes)
        return self.light_net.extract_features(rgb, bitplanes)

    def legacy_state_dict(self, *, keep_vars: bool = False) -> OrderedDict:
        """仅历史卷积结构使用原始 model_new 的键名导出全部状态。"""
        if self.enhancement_backbone != "legacy_cnn":
            raise RuntimeError("旧格式导出仅支持 enhancement_backbone=legacy_cnn；DEA 请使用 state_dict()")
        current = super().state_dict(keep_vars=keep_vars)
        legacy = OrderedDict((_legacy_key(key), value) for key, value in current.items())
        metadata = getattr(current, "_metadata", None)
        if metadata is not None:
            legacy._metadata = OrderedDict()
            for key, value in metadata.items():
                legacy_key = _legacy_key(key + ".").rstrip(".") if key else ""
                if legacy_key == "light_net.bit_encoder":
                    legacy_key = "light_net"
                legacy._metadata[legacy_key] = value
        return legacy

    def load_legacy_state_dict(self, state_dict: Mapping,
                               strict: bool = True, assign: bool = False):
        """加载原始 model_new 格式的权重，完整校验键名与张量形状。"""
        if self.enhancement_backbone != "legacy_cnn":
            raise RuntimeError("旧 CNN 权重需要 enhancement_backbone=legacy_cnn，不能加载到 DEA 主干")
        if not strict:
            raise ValueError("SIMI checkpoint loading requires strict=True")
        current = super().state_dict()
        legacy = self.legacy_state_dict()
        _check_state_dict(state_dict, legacy)
        translated = OrderedDict(
            (key, state_dict[_legacy_key(key)]) for key in current
        )
        return super().load_state_dict(translated, strict=True, assign=assign)

    def load_state_dict(self, state_dict: Mapping,
                        strict: bool = True, assign: bool = False):
        """严格加载新格式权重，或使用完整原始模型键名的旧格式权重。"""
        if not strict:
            raise ValueError("SIMI checkpoint loading requires strict=True")
        has_dea_projection = "light_net.curve_network.input_projection.weight" in state_dict
        has_cnn_stem = ("light_net.e_conv1.weight" in state_dict
                        or "light_net.curve_network.e_conv1.weight" in state_dict)
        if self.enhancement_backbone == "dea" and has_cnn_stem:
            raise RuntimeError("旧 CNN 权重需要 enhancement_backbone=legacy_cnn，不能加载到 DEA 主干")
        if self.enhancement_backbone == "legacy_cnn" and has_dea_projection:
            raise RuntimeError("DEA 权重需要 enhancement_backbone=dea，不能加载到旧 CNN 主干")
        current = super().state_dict()
        if (self.enhancement_backbone == "legacy_cnn"
                and set(state_dict) == set(self.legacy_state_dict())):
            return self.load_legacy_state_dict(state_dict, strict=True, assign=assign)
        _check_state_dict(state_dict, current)
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def checkpoint_config(self) -> dict[str, str]:
        """记录影响结构和输入语义的配置，避免推理误用研究检查点。"""
        return {
            "bit_mode": self.bit_mode,
            "enhancement_guidance": self.enhancement_guidance,
            "enhancement_backbone": self.enhancement_backbone,
        }

    def validate_checkpoint_config(self, checkpoint: Mapping) -> None:
        """新检查点明确校验配置；无元数据的旧权重由完整状态校验兜底。"""
        saved = checkpoint.get("model_config")
        if saved is None:
            return
        if not isinstance(saved, Mapping):
            raise RuntimeError("检查点 model_config 必须是映射")
        expected = self.checkpoint_config()
        for key, value in expected.items():
            if saved.get(key) != value:
                raise RuntimeError(
                    f"检查点要求 {key}={saved.get(key)!s}，当前配置为 {value!s}；请显式匹配模型配置"
                )
