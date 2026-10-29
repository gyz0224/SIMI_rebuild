"""保留现有代码的损失公式和算术运算顺序。"""

import torch
from torch import nn
from pytorch_msssim import ssim


class GradientLoss(nn.Module):
    def forward(self, x1, x2):
        batch_size = x1.size()[0]
        h_x = x1.size()[2]
        w_x = x1.size()[3]
        count_h = (x1.size()[2] - 1) * x1.size()[3]
        count_w = x1.size()[2] * (x1.size()[3] - 1)
        h_g1 = x1[:, :, 1:, :] - x1[:, :, :h_x - 1, :]
        h_g2 = x2[:, :, 1:, :] - x2[:, :, :h_x - 1, :]
        w_g1 = x1[:, :, :, 1:] - x1[:, :, :, :w_x - 1]
        w_g2 = x2[:, :, :, 1:] - x2[:, :, :, :w_x - 1]
        h_tv = torch.pow((h_g1 - h_g2), 2).sum()
        w_tv = torch.pow((w_g1 - w_g2), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class LocalColorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(4)

    def forward(self, x, org):
        mean_rgb = self.pool(org)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        mean_rgb1 = self.pool(x)
        mr1, mg1, mb1 = torch.split(mean_rgb1, 1, dim=1)
        r = mr / (mr + mg + mb + 0.0001)
        g = mg / (mr + mg + mb + 0.0001)
        b = mb / (mr + mg + mb + 0.0001)
        r1 = mr1 / (mr1 + mg1 + mb1 + 0.0001)
        g1 = mg1 / (mr1 + mg1 + mb1 + 0.0001)
        b1 = mb1 / (mr1 + mg1 + mb1 + 0.0001)
        return torch.pow(r - r1, 2) + torch.pow(g - g1, 2) + torch.pow(b - b1, 2)


class GlobalColorLoss(nn.Module):
    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        r = mr / (mr + mg + mb + 0.0001)
        g = mg / (mr + mg + mb + 0.0001)
        b = mb / (mr + mg + mb + 0.0001)
        return torch.pow(r - 1/3, 2) + torch.pow(g - 1/3, 2) + torch.pow(b - 1/3, 2)


class SmoothnessLoss(nn.Module):
    def forward(self, x1):
        batch_size = x1.size()[0]
        h_x = x1.size()[2]
        w_x = x1.size()[3]
        count_h = (x1.size()[2] - 1) * x1.size()[3]
        count_w = x1.size()[2] * (x1.size()[3] - 1)
        h_tv = torch.pow((x1[:, :, 1:, :] - x1[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x1[:, :, :, 1:] - x1[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class ExposureLoss(nn.Module):
    def __init__(self, patch_size=4, mean_val=0.8):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x, org):
        mean0 = self.pool(org)
        mr0, mg0, mb0 = torch.split(mean0, 1, dim=1)
        xs0 = mr0 + mg0 + mb0 + 0.0001
        xv0 = 1 - torch.pow(torch.pow((mr0/xs0)-1/3, 2) + torch.pow((mg0/xs0)-1/3, 2) + torch.pow((mb0/xs0)-1/3, 2), 0.5)
        xv0 = 3 * xv0
        mean = self.pool(x)
        mr, mg, mb = torch.split(mean, 1, dim=1)
        xv = mr + mg + mb
        dp = xv - self.mean_val * xv0
        return torch.pow(dp, 2)


class UnsupervisedLoss(nn.Module):
    """组合原有损失目标，并保留各项配置权重。

    默认 SSIM 取值范围为 1，与归一化 RGB 图像一致。
    复现旧源码时可在 YAML 中显式设置 ssim_data_range: 255.0；
    其余损失公式、权重及算术运算顺序保持不变。
    """

    def __init__(
        self,
        patch_size=4,
        brightness=0.8,
        smooth_a_weight=1000.0,
        smooth_b_weight=5000.0,
        exposure_weight=5.0,
        local_color_weight=1000.0,
        global_color_weight=1500.0,
        denoising_ssim_weight=10.0,
        denoising_gradient_weight=40.0,
        denoising_smooth_weight=1.0,
        denoising_scale=10.0,
        ssim_data_range=1.0,
    ):
        super().__init__()
        self.global_color = GlobalColorLoss()
        self.exposure = ExposureLoss(patch_size, brightness)
        self.smoothness = SmoothnessLoss()
        self.local_color = LocalColorLoss()
        self.gradient = GradientLoss()
        self.smooth_a_weight = smooth_a_weight
        self.smooth_b_weight = smooth_b_weight
        self.exposure_weight = exposure_weight
        self.local_color_weight = local_color_weight
        self.global_color_weight = global_color_weight
        self.denoising_ssim_weight = denoising_ssim_weight
        self.denoising_gradient_weight = denoising_gradient_weight
        self.denoising_smooth_weight = denoising_smooth_weight
        self.denoising_scale = denoising_scale
        self.ssim_data_range = ssim_data_range

    def enhancement(self, enhanced, original, a, b):
        smoothness = self.smooth_a_weight * self.smoothness(a) + self.smooth_b_weight * self.smoothness(b)
        exposure = self.exposure_weight * torch.mean(self.exposure(enhanced, original))
        local_color = self.local_color_weight * torch.mean(self.local_color(enhanced, original))
        global_color = self.global_color_weight * torch.mean(self.global_color(enhanced))
        total = exposure + smoothness + local_color + global_color
        return {
            "total": total,
            "smoothness": smoothness,
            "exposure": exposure,
            "local_color": local_color,
            "global_color": global_color,
        }

    def denoising(self, final, enhanced_detached):
        ssim_loss = self.denoising_ssim_weight * torch.mean(
            1 - ssim(final, enhanced_detached, data_range=self.ssim_data_range)
        )
        gradient = self.denoising_gradient_weight * torch.mean(self.gradient(final, enhanced_detached))
        structural = ssim_loss + gradient
        smoothness = self.denoising_smooth_weight * torch.mean(self.smoothness(final))
        total = self.denoising_scale * (structural + smoothness)
        return {"total": total, "ssim": ssim_loss, "gradient": gradient, "smoothness": smoothness}
