"""转换 RGB 图像，保留原实现先以双精度除法归一化、再转为 FP32 的行为。"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image as torchvision_save_image


def read_rgb_array(path: Union[str, Path]) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HWC uint8 RGB image")
    # NumPy 先以 float64 将 uint8 数据除以 255.0，再由 PyTorch 转为 FP32。
    return torch.from_numpy(image / 255.0).float().permute(2, 0, 1)


def load_image(path: Union[str, Path]) -> torch.Tensor:
    return image_to_tensor(read_rgb_array(path))


def save_image(image: torch.Tensor, path: Union[str, Path]) -> Path:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Saving an image requires a single image, not a batch")
        image = image.squeeze(0)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a CHW RGB tensor")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用 torchvision 保留原实现的数值截断和加 0.5 后取整行为。
    torchvision_save_image(image.detach().cpu(), str(path))
    return path
