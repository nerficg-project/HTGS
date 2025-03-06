# -- coding: utf-8 --

"""HTGS/Loss.py: Loss function."""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.FusedDSSIM import fused_dssim


class HTGSLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList) -> None:
        super().__init__()
        self.addLossMetric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.addLossMetric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.addQualityMetric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward({
            'L1_Color': {'input': input, 'target': target},
            'DSSIM_Color': {'input': input, 'target': target},
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })
