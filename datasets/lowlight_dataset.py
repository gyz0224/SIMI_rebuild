"""沿用场景子目录组织方式的非配对低光训练数据集。"""

import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .bitplane import decompose_bitplanes
from utils.image_utils import image_to_tensor


def populate_train_list(root, legacy_order=False):
    image_list = glob.glob(os.path.join(os.fspath(root), "*/*.jpg"))
    if legacy_order:
        # 只有显式历史实验使用外部管理的 Python 随机状态。
        random.shuffle(image_list)
    else:
        # 默认先确定文件顺序，训练采样器独立控制 shuffle 和随机种子。
        image_list.sort()
    return image_list


class LowLightDataset(Dataset):
    """读取非配对低光 RGB，可选择同时返回原始位平面。

    图像先在原始 uint8 表示下缩放，再提取位平面，
    与现有训练数据加载器完全一致。
    仅沿用一层场景子目录和小写 JPG 扩展名的匹配方式；
    默认排序，由训练采样器控制打乱；历史列表顺序需显式选择 legacy_order。
    return_bitplanes 默认保持原来的 (RGB, 位平面) 接口；设为 False 时
    仅返回 RGB，由训练器在训练设备上执行同样的二进制分解。
    """

    def __init__(self, root, image_size=512, return_bitplanes=True, *, legacy_order=False):
        self.train_list = populate_train_list(root, legacy_order=legacy_order)
        self.size = image_size
        self.return_bitplanes = return_bitplanes
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        with Image.open(self.data_list[index]) as image:
            resized = image.convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS)
            image_array = np.array(resized)
            rgb = image_to_tensor(image_array)
            if not self.return_bitplanes:
                return rgb
            image_uint8 = torch.from_numpy(image_array).permute(2, 0, 1)
            bitplanes = decompose_bitplanes(image_uint8)
        return rgb, bitplanes

    def __len__(self):
        return len(self.data_list)


lowlight_loader = LowLightDataset
