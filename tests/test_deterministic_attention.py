"""验证严格 CUDA 注意力的有限输入等价性及完整训练反向兼容性。"""

from contextlib import contextmanager
from copy import deepcopy
import os
import unittest

import torch
from torch.nn import functional as F

from losses import UnsupervisedLoss
from models import SIMI
from models.attention import CBAM


@contextmanager
def deterministic_mode(enabled):
    """临时切换严格模式，同时保持卷积算法选择稳定并恢复全局设置。"""
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


def original_cbam(features, module):
    """独立调用原来的池化模块和注意力公式，不使用修改后的 forward。"""
    maximum = module.mlp(module.max_pool(features))
    average = module.mlp(module.avg_pool(features))
    features = module.sigmoid(maximum + average) * features
    maximum = features.max(dim=1, keepdim=True).values
    average = features.mean(dim=1, keepdim=True)
    attention = module.sigmoid(module.conv(torch.cat([maximum, average], dim=1)))
    return attention * features


class DeterministicAttentionTests(unittest.TestCase):
    """检查并列最大值的梯度、历史关闭路径，以及两种完整模型。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_global_maximum_values_and_tie_gradients_are_exact(self):
        """CPU/CUDA 单双精度的随机值、并列值、零及正负常量均逐值一致。"""
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        shapes = [(1, 1, 1, 1), (2, 5, 4, 7), (2, 7, 33, 35), (1, 3, 64, 65)]
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                for shape in shapes:
                    generator = torch.Generator().manual_seed(4321)
                    cases = [
                        torch.randn(shape, generator=generator),
                        torch.randint(-2, 3, shape, generator=generator).float(),
                        torch.zeros(shape), torch.full(shape, 3.5), torch.full(shape, -3.5),
                    ]
                    for index, values in enumerate(cases):
                        with self.subTest(device=device, dtype=dtype, shape=shape, case=index):
                            reference_input = values.to(device=device, dtype=dtype).requires_grad_()
                            actual_input = reference_input.detach().clone().requires_grad_()
                            weight = torch.randn(shape[:2] + (1, 1), generator=generator).to(
                                device=device, dtype=dtype,
                            )
                            with deterministic_mode(False):
                                expected = F.adaptive_max_pool2d(reference_input, 1)
                                (expected * weight).sum().backward()
                            with deterministic_mode(True):
                                actual = actual_input.flatten(2).max(dim=2).values.reshape(
                                    shape[:2] + (1, 1),
                                )
                                (actual * weight).sum().backward()
                            self.assertTrue(torch.equal(actual, expected))
                            self.assertTrue(torch.equal(actual_input.grad, reference_input.grad))

    def compare_cbam(self, device, dtype, enabled, values):
        """对照原公式，检查输出、输入梯度和全部参数梯度及模块调用路径。"""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(139)
            actual_module = CBAM(8).to(device=device, dtype=dtype)
        reference_module = deepcopy(actual_module)
        actual_input = values.to(device=device, dtype=dtype).requires_grad_()
        reference_input = actual_input.detach().clone().requires_grad_()
        calls = []
        handle = actual_module.max_pool.register_forward_hook(lambda *args: calls.append(True))
        with deterministic_mode(False):
            expected = original_cbam(reference_input, reference_module)
            (expected.square().mean() + expected.mean()).backward()
        with deterministic_mode(enabled):
            actual = actual_module(actual_input)
            (actual.square().mean() + actual.mean()).backward()
        handle.remove()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_input.grad, reference_input.grad))
        self.assertEqual(len(calls), 0 if enabled and device == "cuda" else 1)
        self.assertIsInstance(actual_module.max_pool, torch.nn.AdaptiveMaxPool2d)
        self.assertEqual(list(actual_module.state_dict()), list(reference_module.state_dict()))
        for (name, parameter), (expected_name, expected_parameter) in zip(
            actual_module.named_parameters(), reference_module.named_parameters(),
        ):
            self.assertEqual(name, expected_name)
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.equal(parameter.grad, expected_parameter.grad), name)

    def test_cpu_strict_and_original_disabled_paths_are_exact(self):
        """CPU 严格模式及关闭模式均继续调用原池化，保持原公式全部梯度。"""
        generator = torch.Generator().manual_seed(531)
        for dtype in (torch.float32, torch.float64):
            for enabled in (False, True):
                with self.subTest(dtype=dtype, deterministic=enabled):
                    self.compare_cbam("cpu", dtype, enabled, torch.randn(2, 8, 7, 9, generator=generator))

    @unittest.skipUnless(torch.cuda.is_available(), "需要可用 CUDA 设备")
    def test_cuda_strict_and_original_disabled_paths_are_exact(self):
        """严格 CUDA 的完整注意力及关闭模式与原公式全部输出和梯度一致。"""
        generator = torch.Generator().manual_seed(531)
        cases = [
            torch.randn(2, 8, 9, 7, generator=generator).transpose(-1, -2),
            torch.randint(-2, 3, (2, 8, 7, 9), generator=generator).float(),
            torch.zeros(2, 8, 7, 9), torch.full((2, 8, 7, 9), 3.5),
        ]
        for dtype in (torch.float32, torch.float64):
            for enabled in (False, True):
                for index, values in enumerate(cases):
                    with self.subTest(dtype=dtype, deterministic=enabled, case=index):
                        self.compare_cbam("cuda", dtype, enabled, values)

    @unittest.skipUnless(torch.cuda.is_available(), "需要可用 CUDA 设备")
    def test_strict_cuda_rejects_nonfinite_features(self):
        """严格 CUDA 明确拒绝 NaN 和正负无穷，避免非法值的选择规则差异。"""
        module = CBAM(8).cuda()
        with deterministic_mode(True):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(value=value):
                    features = torch.zeros(1, 8, 4, 4, device="cuda")
                    features[0, 0, 0, 0] = value
                    with self.assertRaisesRegex(ValueError, "只接受有限数值"):
                        module(features)

    @unittest.skipUnless(torch.cuda.is_available(), "需要可用 CUDA 设备")
    def test_full_simi_joint_losses_backward_on_strict_cuda(self):
        """DEA 和历史主干在严格 CUDA 下执行增强与去噪真实损失的完整反向。"""
        with deterministic_mode(True), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for backbone in ("dea", "legacy_cnn"):
                with self.subTest(backbone=backbone):
                    torch.manual_seed(20261003)
                    model = SIMI(enhancement_backbone=backbone, activation_checkpointing=True).cuda().train()
                    losses = UnsupervisedLoss().cuda()
                    rgb = torch.rand(1, 3, 16, 16, device="cuda")
                    enhanced, curve_a, curve_b, denoised = model(rgb)
                    total = losses.enhancement(enhanced, rgb, curve_a, curve_b)["total"]
                    total = total + losses.denoising(denoised, enhanced.detach())["total"]
                    total.backward()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.isfinite(total))
                    self.assertTrue(all(
                        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    ))
                    self.assertIsNotNone(model.light_net.curve_network.e_conv1_a.weight.grad)
                    self.assertTrue(any(parameter.grad is not None for parameter in model.denoise.parameters()))


if __name__ == "__main__":
    unittest.main()
