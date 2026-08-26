from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import center_of_mass, label as connected_components

from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform


def _gaussian_kernel_2d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax, indexing='xy')
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)


def _paste_max(canvas: np.ndarray, kernel: np.ndarray, center_y: int, center_x: int) -> None:
    """canvas = elementwise max(canvas, kernel) centered at (center_y, center_x), clipped to canvas bounds."""
    radius_y, radius_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    h, w = canvas.shape

    y0, y1 = center_y - radius_y, center_y - radius_y + kernel.shape[0]
    x0, x1 = center_x - radius_x, center_x - radius_x + kernel.shape[1]

    ky0, ky1 = max(0, -y0), kernel.shape[0] - max(0, y1 - h)
    kx0, kx1 = max(0, -x0), kernel.shape[1] - max(0, x1 - w)
    cy0, cy1 = max(0, y0), min(h, y1)
    cx0, cx1 = max(0, x0), min(w, x1)

    if ky1 <= ky0 or kx1 <= kx0:
        return

    np.maximum(canvas[cy0:cy1, cx0:cx1], kernel[ky0:ky1,
               kx0:kx1], out=canvas[cy0:cy1, cx0:cx1])


class ConvertSegToMultiChannelHeatmap(SegOnlyTransform):
    """
    Converts an integer blob-label segmentation (1, H, W), values 0..num_classes, into a dense
    (num_classes + 1, H, W) float32 Gaussian-heatmap target: channel 0 is the "combined" heatmap
    (max over all classes), channels 1..num_classes are per-class heatmaps.

    Connected components are found independently per class mask. Any number of same-class instances 
    in one image are each recovered as their own heatmap peak.

    Meant to be appended as the last step of nnUNetTrainer.get_training_transforms /
    get_validation_transforms (see nnUNetTrainerHeatmapMSE), after spatial augmentation has already
    run on the still-discrete blob map.
    """

    def __init__(self, num_classes: int, sigma: float = 3.0):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self._kernel = _gaussian_kernel_2d(sigma)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        assert segmentation.ndim == 3 and segmentation.shape[0] == 1, \
            f"ConvertSegToMultiChannelHeatmap expects a (1, H, W) segmentation, got {tuple(segmentation.shape)}"
        seg = segmentation[0].numpy()
        h, w = seg.shape

        heatmap = np.zeros((self.num_classes + 1, h, w), dtype=np.float32)

        for c in range(1, self.num_classes + 1):
            mask = seg == c
            if not mask.any():
                continue
            labeled, num_components = connected_components(mask)
            if num_components == 0:
                continue
            centers = center_of_mass(
                mask, labeled, range(1, num_components + 1))
            for cy, cx in centers:
                _paste_max(heatmap[c], self._kernel,
                           int(round(cy)), int(round(cx)))

        if self.num_classes > 0:
            heatmap[0] = heatmap[1:].max(axis=0)

        return torch.from_numpy(heatmap)


class ConvertSegToSingleChannelHeatmap(SegOnlyTransform):
    """
    Single-label counterpart to ConvertSegToMultiChannelHeatmap, for datasets with exactly one
    foreground label where all landmark types have been merged into 
    (see JunctionDetection/PreProcessing/create_nnunet_dataset_variants.py). 
    Converts an integer blob-label segmentation (1, H, W), values 0/1, into a single-channel (1, H, W) 
    float32 Gaussian-heatmap target.

    Meant to be appended as the last step of nnUNetTrainer.get_training_transforms /
    get_validation_transforms, after spatial augmentation has already run on the still-discrete blob
    map.
    """

    def __init__(self, sigma: float = 3.0):
        super().__init__()
        self.sigma = sigma
        self._kernel = _gaussian_kernel_2d(sigma)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        assert segmentation.ndim == 3 and segmentation.shape[0] == 1, \
            f"ConvertSegToSingleChannelHeatmap expects a (1, H, W) segmentation, got {tuple(segmentation.shape)}"
        seg = segmentation[0].numpy()
        h, w = seg.shape

        heatmap = np.zeros((1, h, w), dtype=np.float32)
        mask = seg > 0
        if mask.any():
            labeled, num_components = connected_components(mask)
            if num_components > 0:
                centers = center_of_mass(
                    mask, labeled, range(1, num_components + 1))
                for cy, cx in centers:
                    _paste_max(heatmap[0], self._kernel,
                               int(round(cy)), int(round(cx)))

        return torch.from_numpy(heatmap)
