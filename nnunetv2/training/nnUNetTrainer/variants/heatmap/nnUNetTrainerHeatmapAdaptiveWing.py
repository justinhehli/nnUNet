import os

import torch

from nnunetv2.training.loss.adaptive_wing import AdaptiveWingLoss, WeightedAdaptiveWingLoss
from nnunetv2.training.nnUNetTrainer.variants.heatmap.nnUNetTrainerHeatmapMSE import nnUNetTrainerHeatmapMSE


class nnUNetTrainerHeatmapAdaptiveWing(nnUNetTrainerHeatmapMSE):
    """
    Same multi-instance, multi-channel Gaussian-heatmap regression setup as nnUNetTrainerHeatmapMSE,
    but trained with the Adaptive Wing loss see nnunetv2.training.loss.adaptive_wing for the loss.

    reads the following additional env vars (on top of nnUNetTrainerHeatmapMSE's
    NNUNET_HEATMAP_SIGMA/THRESHOLD/MIN_DISTANCE):
        NNUNET_AWING_OMEGA               paper's omega                                  (default 14.0)
        NNUNET_AWING_THETA               paper's theta                                  (default 0.5)
        NNUNET_AWING_EPSILON             paper's epsilon                                (default 1.0)
        NNUNET_AWING_ALPHA               paper's alpha                                  (default 2.1, must be > 2)
        NNUNET_AWING_USE_WEIGHT_MAP      apply the Weighted Loss Map (Sec. 4.3)         (default true)
        NNUNET_AWING_WEIGHT              paper's W for the Weighted Loss Map            (default 10.0)
        NNUNET_AWING_DILATION_THRESHOLD  paper's threshold for the dilated-heatmap mask (default 0.2)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.awing_omega = float(os.environ.get("NNUNET_AWING_OMEGA", 14.0))
        self.awing_theta = float(os.environ.get("NNUNET_AWING_THETA", 0.5))
        self.awing_epsilon = float(os.environ.get("NNUNET_AWING_EPSILON", 1.0))
        self.awing_alpha = float(os.environ.get("NNUNET_AWING_ALPHA", 2.1))
        self.awing_use_weight_map = os.environ.get(
            "NNUNET_AWING_USE_WEIGHT_MAP", "true").lower() == "true"
        self.awing_weight = float(os.environ.get("NNUNET_AWING_WEIGHT", 10.0))
        self.awing_dilation_threshold = float(
            os.environ.get("NNUNET_AWING_DILATION_THRESHOLD", 0.2))

        self.print_to_log_file(
            f"nnUNetTrainerHeatmapAdaptiveWing: omega={self.awing_omega}, theta={self.awing_theta}, "
            f"epsilon={self.awing_epsilon}, alpha={self.awing_alpha}, "
            f"use_weight_map={self.awing_use_weight_map}"
            + (f", weight={self.awing_weight}, dilation_threshold={self.awing_dilation_threshold}"
               if self.awing_use_weight_map else "")
        )

    def _build_loss(self):
        if self.awing_use_weight_map:
            return WeightedAdaptiveWingLoss(
                omega=self.awing_omega, theta=self.awing_theta, epsilon=self.awing_epsilon,
                alpha=self.awing_alpha, weight=self.awing_weight,
                dilation_threshold=self.awing_dilation_threshold)
        return AdaptiveWingLoss(
            omega=self.awing_omega, theta=self.awing_theta, epsilon=self.awing_epsilon,
            alpha=self.awing_alpha)
