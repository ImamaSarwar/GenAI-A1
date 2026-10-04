"""L1 + SSIM loss and image-quality metrics. Images: float (B,3,H,W) in [0,1]."""
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def ssim_per_sample(x, y):
    return ssim(x, y, data_range=1.0, size_average=False)      # (B,)


def psnr_per_sample(x, y, eps=1e-10, max_db=50.0):
    """PSNR in dB. Capped at max_db so identical images (clean input vs. clean target)
    do not give inf/100 dB and distort the averages."""
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3))
    return (10 * torch.log10(1.0 / (mse + eps))).clamp(max=max_db)


def recon_loss(pred, target, alpha):
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM).  Returns (loss, l1, ssim)."""
    l1 = F.l1_loss(pred, target)
    s = ssim(pred, target, data_range=1.0, size_average=True)
    return alpha * l1 + (1 - alpha) * (1 - s), l1.detach(), s.detach()
