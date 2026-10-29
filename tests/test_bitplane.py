"""验证位序、原始实现等价性、可插拔接口约定及 CUDA 输出一致性。"""

import unittest

import numpy as np
import torch

from datasets.bitplane import (
    BinaryBitPlaneExtractor,
    BitPlaneExtractor,
    bitplanes_from_rgb,
    decompose_bitplanes,
)


class BitPlaneTests(unittest.TestCase):
    def setUp(self):
        self.extractor = BinaryBitPlaneExtractor()

    def test_all_bytes_and_rgb_channel_order(self):
        # 为 R、G、B 使用不同的数值模式，确保通道顺序错误时无法通过测试。
        values = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
        image = torch.stack([values, 255 - values, values ^ 85]).unsqueeze(0)
        result = self.extractor(image)
        expected = np.unpackbits(image[0].permute(1, 2, 0).numpy(), axis=2)
        expected = torch.from_numpy(expected.transpose(2, 0, 1).copy()).float().unsqueeze(0)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device, image.device)
        weights = (2 ** torch.arange(7, -1, -1)).reshape(1, 1, 8, 1, 1)
        recovered = (result.reshape(1, 3, 8, 16, 16) * weights).sum(2).to(torch.uint8)
        self.assertTrue(torch.equal(recovered, image))

    def test_batch_and_noncontiguous_input(self):
        image = torch.randint(0, 256, (4, 3, 11, 17), dtype=torch.uint8).transpose(2, 3)
        self.assertFalse(image.is_contiguous())
        result = self.extractor(image)
        self.assertEqual(tuple(result.shape), (4, 24, 17, 11))
        individually = torch.cat([self.extractor(sample.unsqueeze(0)) for sample in image])
        self.assertTrue(torch.equal(result, individually))

    def test_normalized_float_and_existing_wrappers(self):
        image = torch.randint(0, 256, (3, 3, 13, 19), dtype=torch.uint8)
        rgb = torch.from_numpy(image.numpy() / 255.0).float()
        expected = self.extractor(image)
        self.assertTrue(torch.equal(self.extractor(rgb), expected))
        self.assertTrue(torch.equal(decompose_bitplanes(image), expected))
        self.assertTrue(torch.equal(bitplanes_from_rgb(rgb), expected))
        self.assertTrue(torch.equal(decompose_bitplanes(image[0]), expected[0]))
        self.assertTrue(torch.equal(bitplanes_from_rgb(rgb[0]), expected[0]))

    def test_abstract_interface_and_parameter_free_binary(self):
        with self.assertRaises(TypeError):
            BitPlaneExtractor()
        self.assertIsInstance(self.extractor, BitPlaneExtractor)
        self.assertEqual(list(self.extractor.parameters()), [])
        self.assertEqual(list(self.extractor.buffers()), [])
        self.assertEqual(len(self.extractor.state_dict()), 0)
        image = torch.rand(1, 3, 5, 7, requires_grad=True)
        self.assertFalse(self.extractor(image).requires_grad)

    def test_reject_wrong_tensor_metadata(self):
        for shape in ((3, 7, 9), (2, 1, 7, 9), (2, 4, 7, 9), (2, 3, 4, 7, 9)):
            with self.assertRaises(ValueError):
                self.extractor(torch.zeros(shape, dtype=torch.uint8))
        with self.assertRaises(TypeError):
            self.extractor(torch.zeros(2, 3, 7, 9, dtype=torch.int32))
        with self.assertRaises(TypeError):
            self.extractor(None)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_batch_matches_cpu(self):
        # 在当前可用的 GPU 上执行实际的批量张量运算。
        image = torch.randint(0, 256, (8, 3, 128, 96), dtype=torch.uint8)
        expected = self.extractor(image)
        extractor = BinaryBitPlaneExtractor().cuda()
        actual = extractor(image.cuda())
        self.assertEqual(actual.device.type, "cuda")
        self.assertTrue(torch.equal(actual.cpu(), expected))
        rgb = image.float().cuda() / 255
        self.assertTrue(torch.equal(extractor(rgb), actual))


if __name__ == "__main__":
    unittest.main()
