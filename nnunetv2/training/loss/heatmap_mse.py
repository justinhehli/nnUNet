import torch
from torch import nn
from torch.nn import functional as F


class SigmoidMSELoss(nn.Module):
    """
    MSE between sigmoid(network_output) and a dense (multi-channel) heatmap target in [0, 1].

    Direct port of nnLandmark's own Nonlin_MSE_loss
    (nnlandmark/training/loss/regression.py, https://github.com/MIC-DKFZ/nnLandmark), used here as
    the requested "MSE loss like in nnLandmark" for heatmap regression.
    """

    def __init__(self):
        super().__init__()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(torch.sigmoid(net_output), target)
