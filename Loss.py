"""HTGS/Loss.py: Loss function."""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import fused_dssim
from Optim.Losses.BackgroundEntropy import background_entropy


def background_separation_loss(alpha: torch.Tensor, mask: torch.Tensor = None, tau: float = 1e-3) -> torch.Tensor:
    """
    Penalize foreground leakage into sky regions.
    Applies ReLU(alpha - tau) only where sky_mask == 1.
    """
    if mask is None:
        return torch.tensor(0.0, device=alpha.device)
    num = torch.relu(alpha - tau) * mask
    den = mask.sum().clamp(min=1.0)
    return num.sum() / den


class HTGSLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList) -> None:
        super().__init__()
        self.add_loss_metric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.add_loss_metric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_loss_metric('BCE_Alpha', background_entropy, loss_config.LAMBDA_BCE)
        self.add_loss_metric('BG_Separation', background_separation_loss, loss_config.LAMBDA_BG_SEPARATION)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, input: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return super().forward({
            'L1_Color': {'input': input, 'target': target},
            'DSSIM_Color': {'input': input, 'target': target},
            'BCE_Alpha': {'input': alpha},
            'BG_Separation': {'alpha': alpha, 'mask': mask},
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })
