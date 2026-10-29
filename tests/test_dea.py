"""独立验证 DEA 卷积、内容引导注意力、参数预测头和架构检查点。"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from datasets import bitplanes_from_rgb
from inference import test as inference_cli
from losses import UnsupervisedLoss
from models import CurveEnhancementNetwork, EnhancementCurveNetwork, SIMI
from models.blocks import DEABlock, DEConv
from train import load_config
from trainers.trainer import Trainer
from utils.image_utils import load_image, save_image


ROOT = Path(__file__).resolve().parents[1]


def convolution(features, layer):
    """直接计算普通卷积，并显式应用其反射填充和分组约定。"""
    padding = layer.padding
    if layer.padding_mode != "zeros":
        height, width = padding
        features = F.pad(features, (width, width, height, height), mode=layer.padding_mode)
        padding = 0
    return F.conv2d(features, layer.weight, layer.bias, layer.stride, padding, layer.dilation, layer.groups)


def reference_deconv(features, layer):
    """独立构造五个差分卷积核，不调用实现中的 get_weight 或 forward。"""
    central = layer.conv1_1.conv.weight
    flattened = central.flatten(2)
    center_mask = central.new_zeros(9)
    center_mask[4] = 1
    central = (flattened - flattened.sum(2, keepdim=True) * center_mask).reshape_as(central)
    horizontal = layer.conv1_2.conv.weight
    horizontal = torch.stack((horizontal, torch.zeros_like(horizontal), -horizontal), dim=-1)
    vertical = layer.conv1_3.conv.weight
    vertical = torch.stack((vertical, torch.zeros_like(vertical), -vertical), dim=-2)
    angular = layer.conv1_4.conv.weight
    angular_flat = angular.flatten(2)
    angular = (angular_flat - angular_flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).reshape_as(angular)
    normal = layer.conv1_5.weight
    weight = central + horizontal + vertical + angular + normal
    bias = layer.conv1_1.conv.bias + layer.conv1_2.conv.bias + layer.conv1_3.conv.bias + layer.conv1_4.conv.bias + layer.conv1_5.bias
    return F.conv2d(features, weight, bias, stride=1, padding=1)


def reference_dea_block(features, layer):
    """使用独立卷积和注意力公式，复现两次残差及通道交错配对。"""
    residual = torch.relu(reference_deconv(features, layer.conv1)) + features
    residual = convolution(residual, layer.conv2)
    pooled = F.adaptive_avg_pool2d(residual, 1)
    channel = convolution(torch.relu(convolution(pooled, layer.ca.ca[0])), layer.ca.ca[2])
    spatial_input = torch.cat((residual.mean(1, keepdim=True), residual.max(1, keepdim=True).values), dim=1)
    spatial = convolution(spatial_input, layer.sa.sa)
    attention = spatial + channel
    paired = torch.stack((residual, attention), dim=2).reshape(residual.shape[0], residual.shape[1] * 2, *residual.shape[2:])
    pixel = torch.sigmoid(convolution(paired, layer.pa.pa2))
    return residual * pixel + features


class DEAReferenceTests(unittest.TestCase):
    """在不同设备和类型上对照独立参考计算以及全部梯度。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def compare(self, constructor, reference, device, dtype):
        """保持相同权重，比较输出、输入梯度和每个参数的梯度。"""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(139)
            module = constructor(8).to(device=device, dtype=dtype)
        expected_module = deepcopy(module)
        generator = torch.Generator().manual_seed(513)
        image = torch.randn((2, 8, 9, 7), generator=generator).transpose(-1, -2).to(device=device, dtype=dtype)
        image = image.detach().requires_grad_()
        reference_image = image.detach().clone().requires_grad_()
        actual = module(image)
        expected = reference(reference_image, expected_module)
        self.assertEqual(actual.device, image.device)
        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(actual.shape, image.shape)
        tolerance = {"rtol": 1e-10, "atol": 1e-11} if dtype == torch.float64 else {"rtol": 1e-5, "atol": 1e-6}
        torch.testing.assert_close(actual, expected, **tolerance)
        (actual.square().mean() + actual.mean()).backward()
        (expected.square().mean() + expected.mean()).backward()
        torch.testing.assert_close(image.grad, reference_image.grad, **tolerance)
        actual_parameters, expected_parameters = list(module.named_parameters()), list(expected_module.named_parameters())
        self.assertEqual(len(actual_parameters), len(expected_parameters))
        for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(actual_parameters, expected_parameters):
            self.assertEqual(actual_name, expected_name)
            self.assertIsNotNone(actual_parameter.grad, actual_name)
            self.assertTrue(torch.isfinite(actual_parameter.grad).all(), actual_name)
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, **tolerance)
        deconv = module if isinstance(module, DEConv) else module.conv1
        for branch in (deconv.conv1_1, deconv.conv1_2, deconv.conv1_3, deconv.conv1_4):
            weight, bias = branch.get_weight()
            self.assertEqual(weight.dtype, dtype)
            self.assertEqual(weight.device, image.device)
            self.assertEqual(bias.dtype, dtype)
            self.assertEqual(bias.device, image.device)

    def test_deconv_and_cga_reference_values_and_gradients_on_cpu(self):
        """CPU 单精度和双精度均与手工 DEConv/CGA 计算一致。"""
        for constructor, reference in ((DEConv, reference_deconv), (DEABlock, reference_dea_block)):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(module=constructor.__name__, dtype=dtype):
                    self.compare(constructor, reference, "cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "需要可用 CUDA 设备")
    def test_deconv_and_cga_reference_values_and_gradients_on_cuda(self):
        """CUDA 单精度和双精度保留设备、类型以及正确的参考梯度。"""
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, benchmark=False, deterministic=True, allow_tf32=torch.backends.cudnn.allow_tf32):
            for constructor, reference in ((DEConv, reference_deconv), (DEABlock, reference_dea_block)):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(module=constructor.__name__, dtype=dtype):
                        self.compare(constructor, reference, "cuda:0", dtype)

    def test_default_backbone_runs_seven_dea_blocks_and_all_independent_heads(self):
        """默认主干运行七个 DEA 块，十四个独立预测头均参与求导。"""
        model = CurveEnhancementNetwork()
        core = model.curve_network
        self.assertEqual(model.enhancement_backbone, "dea")
        self.assertEqual((core.input_projection.in_channels, core.input_projection.out_channels), (27, 32))
        self.assertEqual(sum(isinstance(module, DEABlock) for module in core.modules()), 7)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 377732)
        calls, first_a_logits, first_b_logits, handles = [], [], [], []
        for stage in range(1, 8):
            block = getattr(core, f"e_conv{stage}")
            self.assertIsInstance(block, DEABlock)
            handles.append(block.register_forward_hook(lambda module, inputs, output, stage=stage: calls.append(stage)))
            head_a, head_b = getattr(core, f"e_conv{stage}_a"), getattr(core, f"e_conv{stage}_b")
            self.assertIsNot(head_a, head_b)
            # 将测试预测头置于非饱和区域，检查每个头的梯度连通性。
            with torch.no_grad():
                head_a.weight.zero_()
                head_a.bias.zero_()
                head_b.weight.zero_()
                head_b.bias.zero_()
        handles.extend((
            core.e_conv1_a.register_forward_hook(lambda module, inputs, output: first_a_logits.append(output)),
            core.e_conv1_b.register_forward_hook(lambda module, inputs, output: first_b_logits.append(output)),
        ))
        image = torch.rand((2, 3, 16, 16), generator=torch.Generator().manual_seed(113))
        try:
            enhanced, a, b, features = model.forward_with_aux(image, bitplanes_from_rgb(image))
            (enhanced.mean() + a.mean() + b.mean()).backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, list(range(1, 8)))
        self.assertEqual(len(first_a_logits), 1)
        self.assertEqual(len(first_b_logits), 1)
        self.assertEqual(tuple(features.shape), (2, 32, 16, 16))
        self.assertTrue(torch.all((a >= -1) & (a <= 1)))
        self.assertTrue(torch.all((b >= 0.5) & (b <= 1)))
        for stage in range(1, 8):
            for suffix in ("a", "b"):
                head = getattr(core, f"e_conv{stage}_{suffix}")
                self.assertIsNotNone(head.weight.grad)
                self.assertIsNotNone(head.bias.grad)
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                self.assertTrue(torch.any(head.bias.grad != 0))

    def test_dea_keeps_the_original_seven_curve_updates(self):
        """更换特征主干后，七级曲线数学公式和默认张量接口仍保持原样。"""
        model = CurveEnhancementNetwork()
        image = torch.rand((1, 3, 16, 20), generator=torch.Generator().manual_seed(753))
        planes = bitplanes_from_rgb(image)
        with torch.no_grad():
            enhanced, a, b, _ = model.forward_with_aux(image, planes)
            expected = image
            for stage in range(7):
                expected = expected + a[:, 3 * stage:3 * stage + 3] * 1 / (
                    1 + torch.exp(-10 * (-expected + b[:, 3 * stage:3 * stage + 3] - 0.1))
                ) * expected * (b[:, 3 * stage:3 * stage + 3] - expected) * (1 / b[:, 3 * stage:3 * stage + 3])
            self.assertTrue(torch.equal(enhanced, expected))
            self.assertTrue(torch.equal(model(image, planes), enhanced))
            self.assertTrue(torch.equal(EnhancementCurveNetwork.apply_curves(image, a, b), enhanced))

    def test_checkpoint_profiles_reject_each_other_with_actionable_message(self):
        """两种主干不能互相加载权重，提示中必须给出所需的架构选择。"""
        dea, legacy = SIMI(), SIMI(enhancement_backbone="legacy_cnn")
        self.assertEqual(sum(parameter.numel() for parameter in dea.parameters()), 488044)
        self.assertEqual(sum(parameter.numel() for parameter in legacy.parameters()), 216219)
        for operation in (dea.legacy_state_dict, lambda: dea.load_legacy_state_dict(legacy.legacy_state_dict()), lambda: dea.load_state_dict(legacy.state_dict()), lambda: dea.load_state_dict(legacy.legacy_state_dict())):
            with self.assertRaisesRegex(RuntimeError, "enhancement_backbone=legacy_cnn"):
                operation()
        with self.assertRaisesRegex(RuntimeError, "enhancement_backbone=dea"):
            legacy.load_state_dict(dea.state_dict())

    def compare_activation_checkpointing(self, device):
        """检查重计算开关是否保留输出、梯度和一次 AdamW 更新的结果。"""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(829)
            plain = CurveEnhancementNetwork(activation_checkpointing=False).to(device)
            torch.manual_seed(829)
            recomputed = CurveEnhancementNetwork(activation_checkpointing=True).to(device)
        plain.train()
        recomputed.train()
        image = torch.rand((1, 3, 16, 16), generator=torch.Generator().manual_seed(571)).to(device)
        left_image = image.detach().requires_grad_()
        right_image = image.detach().clone().requires_grad_()
        left_planes = bitplanes_from_rgb(image).detach().requires_grad_()
        right_planes = left_planes.detach().clone().requires_grad_()
        left_outputs = plain.forward_with_aux(left_image, left_planes)
        right_outputs = recomputed.forward_with_aux(right_image, right_planes)
        gradient_differences, gradient_reference_norms = [], []

        def compare_gradient(left, right):
            """依据梯度尺度控制 CUDA 反射填充反传的浮点归约误差。"""
            if device == "cpu":
                self.assertTrue(torch.equal(left, right))
            else:
                # 原损失产生千量级梯度。重复普通反传本身的峰值归一化误差
                # 已测约 8.6e-7，故用 1e-6 的归一化绝对容差。
                scale = left.detach().abs().max().clamp_min(1)
                torch.testing.assert_close(left / scale, right / scale, rtol=1e-5, atol=1e-6)
                gradient_differences.append((left - right).double().square().sum())
                gradient_reference_norms.append(left.double().square().sum())

        for left, right in zip(left_outputs, right_outputs):
            self.assertTrue(torch.equal(left, right))
        objective = UnsupervisedLoss()
        left_loss = objective.enhancement(left_outputs[0], left_image, left_outputs[1], left_outputs[2])["total"]
        right_loss = objective.enhancement(right_outputs[0], right_image, right_outputs[1], right_outputs[2])["total"]
        self.assertTrue(torch.equal(left_loss, right_loss))
        left_loss.backward()
        right_loss.backward()
        compare_gradient(left_image.grad, right_image.grad)
        compare_gradient(left_planes.grad, right_planes.grad)
        for left, right in zip(plain.parameters(), recomputed.parameters()):
            self.assertEqual(left.grad is None, right.grad is None)
            if left.grad is not None:
                compare_gradient(left.grad, right.grad)
        if device != "cpu":
            # 对完整输入与参数梯度向量设置相对 L2 界限，避免单个近零张量
            # 的相对误差被取消效应放大；每个张量仍独立满足上面的峰值界限。
            difference_norm = torch.stack(gradient_differences).sum().sqrt()
            reference_norm = torch.stack(gradient_reference_norms).sum().sqrt().clamp_min(1e-12)
            self.assertLessEqual(float(difference_norm / reference_norm), 1e-6)
        left_optimizer = torch.optim.AdamW(plain.parameters(), lr=1e-5, weight_decay=1e-4)
        right_optimizer = torch.optim.AdamW(recomputed.parameters(), lr=1e-5, weight_decay=1e-4)
        torch.nn.utils.clip_grad_norm_(plain.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(recomputed.parameters(), 0.1)
        left_optimizer.step()
        right_optimizer.step()
        self.assertEqual(list(plain.state_dict()), list(recomputed.state_dict()))
        for key, value in plain.state_dict().items():
            if device == "cpu":
                self.assertTrue(torch.equal(value, recomputed.state_dict()[key]))
            else:
                # 实测一步参数最大绝对差低于 3e-8，使用独立的紧容差。
                torch.testing.assert_close(value, recomputed.state_dict()[key], rtol=1e-6, atol=5e-8)
        left_state, right_state = left_optimizer.state_dict(), right_optimizer.state_dict()
        self.assertEqual(left_state["param_groups"], right_state["param_groups"])
        self.assertEqual(list(left_state["state"]), list(right_state["state"]))
        for index, state in left_state["state"].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    other = right_state["state"][index][key]
                    if device == "cpu" or key == "step":
                        self.assertTrue(torch.equal(value, other))
                    else:
                        # 实测一阶矩差低于 2.4e-10，二阶矩差低于 3.6e-14。
                        absolute_tolerance = 5e-10 if key == "exp_avg" else 5e-14
                        torch.testing.assert_close(value, other, rtol=1e-5, atol=absolute_tolerance)
                else:
                    self.assertEqual(value, right_state["state"][index][key])

    def test_activation_checkpointing_preserves_cpu_outputs_gradients_and_optimizer_step(self):
        """CPU 重计算前后所有数值严格相等，包括 AdamW 状态。"""
        self.compare_activation_checkpointing("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "需要可用 CUDA 设备")
    def test_activation_checkpointing_preserves_cuda_outputs_gradients_and_optimizer_step(self):
        """CUDA 确定性卷积下，重计算前后在明确浮点容差内保持一致。"""
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, benchmark=False, deterministic=True, allow_tf32=False):
            self.compare_activation_checkpointing("cuda:0")

    def test_dea_rejects_spatial_dimensions_too_small_for_reflect_padding(self):
        """DEA 的反射填充需要高和宽大于 3，应在网络入口明确拒绝。"""
        model = CurveEnhancementNetwork()
        for height, width in ((3, 8), (8, 3), (1, 1)):
            with self.subTest(height=height, width=width), self.assertRaises(ValueError):
                model(torch.rand(1, 3, height, width), torch.rand(1, 24, height, width))

    def test_dea_training_and_inference_checkpoint_roundtrip_for_all_modes(self):
        """三种 DEA 训练模式的检查点均可恢复，并用于实际单图推理。"""
        for mode in ("enhancement", "denoising", "joint"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                pixels = np.random.default_rng(95).integers(0, 256, (16, 20, 3), dtype=np.uint8)
                input_path = directory / "input.png"
                Image.fromarray(pixels).save(input_path)
                rgb = load_image(input_path).unsqueeze(0)
                config = load_config(ROOT / "configs/default.yaml")
                config["validation"]["enabled"] = False
                config["runtime"]["device"] = "cpu"
                config["train"]["mode"] = mode
                config["train"]["load_pretrain"] = False
                source = Trainer(config)
                source.train_step(rgb)
                checkpoint = source.save_checkpoint(directory / "dea.pth")
                payload = torch.load(checkpoint, map_location="cpu")
                self.assertTrue({"net", "optimizer_light", "optimizer_denoise", "model_config"}.issubset(payload))
                self.assertEqual(payload["model_config"], {"bit_mode": "binary", "enhancement_guidance": "none", "enhancement_backbone": "dea"})
                self.assertEqual(list(payload["net"]), list(source.model.state_dict()))
                restored = Trainer(config)
                restored.load_checkpoint(checkpoint)
                source.model.eval()
                restored.model.eval()
                with torch.no_grad():
                    expected = source.model(rgb)
                    actual = restored.model(rgb)
                for left, right in zip(actual, expected):
                    self.assertTrue(torch.equal(left, right))
                output_path, expected_path = directory / "output.png", directory / "expected.png"
                args = inference_cli.build_parser().parse_args(["--checkpoint", str(checkpoint), "--input", str(input_path), "--output", str(output_path), "--device", "cpu", "--stage", "enhancement", "--no-benchmark"])
                with redirect_stdout(io.StringIO()):
                    inference_cli.run_inference(args)
                save_image(expected[0], expected_path)
                with Image.open(output_path) as actual_image, Image.open(expected_path) as expected_image:
                    self.assertTrue(np.array_equal(np.array(actual_image), np.array(expected_image)))
                incompatible = deepcopy(config)
                incompatible["enhancement_backbone"] = "legacy_cnn"
                with self.assertRaisesRegex(RuntimeError, "enhancement_backbone"):
                    Trainer(incompatible).load_checkpoint(checkpoint, resume_optimizers=False)


if __name__ == "__main__":
    unittest.main()
