"""按照原实现，在 [0, 1] 数值范围内计算有参考指标 PSNR 和 SSIM。"""

from collections.abc import Iterable, Mapping

import torch
from pytorch_msssim import SSIM


def _as_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected a CHW or NCHW RGB tensor")
    return image


def compute_psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    prediction, target = _as_batch(prediction), _as_batch(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and ground truth must have the same dimensions")
    mse = torch.mean((prediction - target) ** 2)
    return 20 * torch.log10(data_range / torch.sqrt(mse))


def compute_ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    prediction, target = _as_batch(prediction), _as_batch(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and ground truth must have the same dimensions")
    metric = SSIM(data_range=data_range, size_average=True, channel=3).to(
        device=prediction.device, dtype=prediction.dtype
    )
    return metric(prediction, target)


def image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "psnr": compute_psnr(prediction, target).detach(),
        "ssim": compute_ssim(prediction, target).detach(),
    }


def average_metrics(results: Iterable[Mapping]) -> dict:
    results = list(results)
    if not results:
        raise ValueError("Cannot average metrics for an empty image list")
    return {
        name: sum(float(result[name]) for result in results) / len(results)
        for name in ("psnr", "ssim")
    }
