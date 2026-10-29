"""验证独立曲线增强接口、七阶段递推及其与原 SIMI 的兼容性。"""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

from datasets import bitplanes_from_rgb
from models import CurveEnhancementNetwork, EnhancementCurveNetwork, EnhancementNetwork


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = (
    ROOT.parent / "SIMI" / "model_new.py",
    ROOT.parent / "model" / "SIMI" / "model_new.py",
)
SOURCE_MODEL = next((path for path in SOURCE_CANDIDATES if path.exists()), SOURCE_CANDIDATES[0])


def import_reference_model():
    """只读加载相邻目录的原始模型，不生成其字节码缓存。"""
    spec = importlib.util.spec_from_file_location("_enhancement_reference", SOURCE_MODEL)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def original_key(key):
    """将独立增强网络的参数名称映射到原始 light_net 名称。"""
    prefixes = (
        ("curve_network.", ""),
        ("bit_encoder.smoothing.", "Conv."),
        ("bit_encoder.fusion.", "Conv1."),
        ("bit_encoder.attention.", "CBAMLayer."),
    )
    for current, original in prefixes:
        if key.startswith(current):
            return original + key[len(current):]
    return key


class EnhancementInterfaceTests(unittest.TestCase):
    """检查独立输出、扩展入口、激活范围和逐级递推约定。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.reference = import_reference_model() if SOURCE_MODEL.exists() else None

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_inputs(self):
        """使用非正方形小批次，覆盖批次维及空间维保留行为。"""
        generator = torch.Generator().manual_seed(413)
        image = torch.rand((2, 3, 13, 17), generator=generator)
        return image, bitplanes_from_rgb(image)

    def initialized_models(self):
        """让原模型、独立接口和兼容接口从完全相同的随机状态初始化。"""
        if self.reference is None:
            self.skipTest("需要相邻目录中的原始 SIMI 模型进行对照")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(271)
            original = self.reference.light_net()
            torch.manual_seed(271)
            standalone = CurveEnhancementNetwork(enhancement_backbone="legacy_cnn")
            torch.manual_seed(271)
            compatible = EnhancementNetwork(enhancement_backbone="legacy_cnn")
        return original, standalone, compatible

    def assert_exact(self, actual, expected, message=""):
        """同时检查张量形状、类型、设备和逐元素数值。"""
        self.assertEqual(actual.shape, expected.shape, message)
        self.assertEqual(actual.dtype, expected.dtype, message)
        self.assertEqual(actual.device, expected.device, message)
        self.assertTrue(torch.equal(actual, expected), message)

    def test_initialization_and_parameter_registration_order(self):
        """独立接口与兼容接口不增加参数，也不改变初始化或注册顺序。"""
        original, standalone, compatible = self.initialized_models()
        expected_state = original.state_dict()
        expected_parameters = list(original.named_parameters())
        for model in (standalone, compatible):
            with self.subTest(model=type(model).__name__):
                state = {original_key(key): value for key, value in model.state_dict().items()}
                self.assertEqual(list(state), list(expected_state))
                for key in expected_state:
                    self.assert_exact(state[key], expected_state[key], key)
                parameters = list(model.named_parameters())
                self.assertEqual(len(parameters), len(expected_parameters))
                for (actual_name, actual), (expected_name, expected) in zip(parameters, expected_parameters):
                    self.assertEqual(original_key(actual_name), expected_name)
                    self.assert_exact(actual, expected, expected_name)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 105907)

    def test_tensor_forward_and_auxiliary_outputs_match_original(self):
        """新接口返回增强张量，辅助输出及兼容接口保留原四项结果。"""
        original, standalone, compatible = self.initialized_models()
        image, planes = self.make_inputs()
        with torch.no_grad():
            expected = original(image, planes)
            output = standalone(image, planes)
            explicit_none = standalone(image, planes, extra_guidance=None)
            auxiliary = standalone.forward_with_aux(image, planes)
            auxiliary_none = standalone.forward_with_aux(image, planes, extra_guidance=None)
            legacy = compatible(image, planes)
            legacy_none = compatible(image, planes, extra_guidance=None)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (2, 3, 13, 17))
        self.assert_exact(output, expected[0])
        self.assert_exact(explicit_none, output)
        for result in (auxiliary, auxiliary_none, legacy, legacy_none):
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 4)
            for index, (actual, reference) in enumerate(zip(result, expected)):
                self.assert_exact(actual, reference, str(index))

    def test_raw_bitplanes_enter_existing_smoothing_once(self):
        """外部位平面作为原始输入进入原平滑模块，不预先重复编码。"""
        model = CurveEnhancementNetwork()
        image, planes = self.make_inputs()
        observed = []
        handle = model.bit_encoder.smoothing[0].register_forward_pre_hook(
            lambda module, inputs: observed.append(inputs[0])
        )
        try:
            with torch.no_grad():
                model(image, planes)
        finally:
            handle.remove()
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0], planes)

    def test_all_stage_activation_ranges_and_first_b_head(self):
        """在历史 legacy_cnn 模式记录七级激活，验证第一组 B 复用 A 预测头。"""
        model = CurveEnhancementNetwork(enhancement_backbone="legacy_cnn")
        core = model.curve_network
        observed_a, observed_b, unused_b = [], [], []

        def record_a(module, inputs, output):
            observed_a.append((inputs[0].detach().clone(), output.detach().clone()))

        def record_b(module, inputs, output):
            observed_b.append((inputs[0].detach().clone(), output.detach().clone()))

        handles = [
            core.tanh.register_forward_hook(record_a),
            core.sigmoid.register_forward_hook(record_b),
            core.e_conv1_b.register_forward_hook(lambda module, inputs, output: unused_b.append(output)),
        ]
        generator = torch.Generator().manual_seed(577)
        fused = torch.randn((2, 27, 13, 17), generator=generator)
        try:
            with torch.no_grad():
                a, b, features = core.predict_curves(fused)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(observed_a), 7)
        self.assertEqual(len(observed_b), 7)
        self.assertEqual(unused_b, [])
        self.assertEqual(tuple(features.shape), (2, 32, 13, 17))
        self.assertEqual(tuple(a.shape), (2, 21, 13, 17))
        self.assertEqual(tuple(b.shape), (2, 21, 13, 17))
        for stage in range(7):
            with self.subTest(stage=stage):
                a_stage = a[:, stage * 3:(stage + 1) * 3]
                b_stage = b[:, stage * 3:(stage + 1) * 3]
                self.assertTrue(torch.all((a_stage >= -1) & (a_stage <= 1)))
                self.assertTrue(torch.all((b_stage >= 0.5) & (b_stage <= 1)))
                self.assert_exact(a_stage, observed_a[stage][1])
                self.assert_exact(b_stage, observed_b[stage][1] * 0.5 + 0.5)
                self.assert_exact(observed_a[stage][1], torch.tanh(observed_a[stage][0]))
                self.assert_exact(observed_b[stage][1], torch.sigmoid(observed_b[stage][0]))
        self.assert_exact(observed_a[0][0], observed_b[0][0])

    def test_apply_curves_matches_seven_original_updates(self):
        """使用各级不同常量验证七次原始递推及各级参数切片。"""
        image = torch.linspace(0.005, 0.95, 2 * 3 * 5 * 7, dtype=torch.float64).reshape(2, 3, 5, 7)
        a_values = (-0.23, 0.11, 0.29, -0.17, 0.34, 0.07, -0.09)
        b_values = (0.56, 0.60, 0.67, 0.74, 0.81, 0.88, 0.95)
        a = torch.cat([torch.full_like(image, value) for value in a_values], dim=1)
        b = torch.cat([torch.full_like(image, value) for value in b_values], dim=1)
        expected = image.clone()
        six_updates = None
        for stage in range(7):
            # 此处逐字保留原模型的递推运算顺序，作为独立的数值对照。
            expected = expected + a[:, 3 * stage:3 * stage + 3, :, :] * 1 / (
                1 + torch.exp(-10 * (-expected + b[:, 3 * stage:3 * stage + 3, :, :] - 0.1))
            ) * expected * (
                b[:, 3 * stage:3 * stage + 3, :, :] - expected
            ) * (1 / b[:, 3 * stage:3 * stage + 3, :, :])
            if stage == 5:
                six_updates = expected.clone()
        actual = EnhancementCurveNetwork.apply_curves(image, a, b)
        self.assert_exact(actual, expected)
        self.assertFalse(torch.equal(actual, six_updates))

    def test_zero_a_preserves_image(self):
        """所有 A 参数为零时，七次递推应严格保持原图。"""
        image, _ = self.make_inputs()
        a = torch.zeros((2, 21, 13, 17))
        b = torch.linspace(0.5, 1.0, 21).reshape(1, 21, 1, 1).expand_as(a)
        self.assert_exact(EnhancementCurveNetwork.apply_curves(image, a, b), image)

    def test_unsupported_guidance_is_explicitly_rejected(self):
        """默认实现不能静默忽略非空引导信息。"""
        image, planes = self.make_inputs()
        for model in (CurveEnhancementNetwork(), EnhancementNetwork()):
            for operation in (model, model.forward_with_aux):
                with self.subTest(model=type(model).__name__, operation=str(operation)):
                    with self.assertRaises(NotImplementedError):
                        operation(image, planes, extra_guidance=torch.ones(2, 1, 13, 17))

    def test_overridden_feature_preparation_accepts_guidance_without_new_parameters(self):
        """子类可扩展特征准备，引导张量保持梯度且不新增网络参数。"""
        class GuidedEnhancement(CurveEnhancementNetwork):
            """仅用于验证扩展入口的无参数引导示例。"""

            def _prepare_features(self, image, bit_feature, extra_guidance=None):
                fused = super()._prepare_features(image, bit_feature, None)
                self.seen_guidance = extra_guidance
                return fused if extra_guidance is None else fused + extra_guidance

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(181)
            base = CurveEnhancementNetwork()
            torch.manual_seed(181)
            guided = GuidedEnhancement()
        self.assertEqual(list(base.state_dict()), list(guided.state_dict()))
        for key, value in base.state_dict().items():
            self.assert_exact(guided.state_dict()[key], value, key)
        image, planes = self.make_inputs()
        guidance = torch.full((2, 27, 13, 17), 0.025, requires_grad=True)
        output = guided(image, planes, extra_guidance=guidance)
        expected = guided.curve_network(image, guided.bit_encoder(image, planes) + guidance)[0]
        self.assertIs(guided.seen_guidance, guidance)
        self.assert_exact(output, expected)
        self.assertFalse(torch.equal(output, base(image, planes)))
        output.sum().backward()
        self.assertIsNotNone(guidance.grad)
        self.assertTrue(torch.any(guidance.grad != 0))
        self.assertEqual(
            [(name, tuple(parameter.shape)) for name, parameter in base.named_parameters()],
            [(name, tuple(parameter.shape)) for name, parameter in guided.named_parameters()],
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in guided.parameters()),
            sum(parameter.numel() for parameter in base.parameters()),
        )
        self.assertEqual(list(base.state_dict()), list(guided.state_dict()))

    def test_input_shape_dtype_and_device_contract(self):
        """错误的类型、通道数、批次、空间尺寸或设备应在接口处明确报错。"""
        model = CurveEnhancementNetwork()
        image, planes = self.make_inputs()
        cases = (
            (image[0], planes, ValueError),
            (image[:, :2], planes, ValueError),
            (image, planes[0], ValueError),
            (image, planes[:, :23], ValueError),
            (image, planes[:1], ValueError),
            (image, planes[:, :, :-1], ValueError),
            (image, planes.double(), ValueError),
            (image.to(torch.uint8), planes.to(torch.uint8), TypeError),
            (None, planes, TypeError),
            (image, None, TypeError),
            (image, torch.empty(planes.shape, device="meta"), ValueError),
        )
        for index, (rgb, bits, exception) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(exception):
                    model(rgb, bits)

    def compare_output_and_gradients(self, device):
        """同时对照增强输出、RGB/位平面梯度和所有原始参数梯度。"""
        original, standalone, _ = self.initialized_models()
        original, standalone = original.to(device), standalone.to(device)
        image, planes = self.make_inputs()
        old_image = image.to(device).detach().requires_grad_()
        old_planes = planes.to(device).detach().requires_grad_()
        new_image = old_image.detach().clone().requires_grad_()
        new_planes = old_planes.detach().clone().requires_grad_()
        expected = original(old_image, old_planes)[0]
        actual = standalone(new_image, new_planes)
        self.assert_exact(actual, expected)
        expected.sum().backward()
        actual.sum().backward()
        self.assert_exact(new_image.grad, old_image.grad, "RGB 梯度")
        self.assert_exact(new_planes.grad, old_planes.grad, "位平面梯度")
        old_parameters = list(original.named_parameters())
        new_parameters = list(standalone.named_parameters())
        self.assertEqual(len(new_parameters), len(old_parameters))
        for (new_name, new), (old_name, old) in zip(new_parameters, old_parameters):
            self.assertEqual(original_key(new_name), old_name)
            self.assertEqual(new.grad is None, old.grad is None, old_name)
            if old.grad is not None:
                self.assert_exact(new.grad, old.grad, old_name)
        self.assertIsNone(standalone.curve_network.e_conv1_b.weight.grad)

    def test_cpu_output_and_gradients_match_original(self):
        """在 CPU 上验证独立增强网络的输出与梯度完全一致。"""
        self.compare_output_and_gradients("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "需要可用 CUDA 设备")
    def test_cuda_output_and_gradients_match_original(self):
        """在 CUDA 小批次上验证输出与梯度，不启动正式训练。"""
        # 原源码的自适应最大池化反向不支持严格确定性，历史对照仅局部关闭。
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            # 仍启用确定性卷积，排除 GPU 归约顺序造成的末位差异。
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ):
                self.compare_output_and_gradients("cuda:0")
        finally:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


if __name__ == "__main__":
    unittest.main()
