import torch

from nnunetv2.training.nnUNetTrainer.variants.heatmap.nnUNetTrainerHeatmapAdaptiveWingReweighted import \
    nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling


class nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingHighFG(nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling):
    """
    nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling with the three foreground-emphasis knobs
    hardcoded to more aggressive values than their defaults, to push down the false-negative rate
    at the cost of more false positives:
        awing_weight              (AWing weighted loss map's foreground weight W): 10.0  -> 20.0
        awing_focal_gamma         (focal modulating exponent on hard positives):    2.0  -> 4.0
        awing_soft_max_downweight (cap on soft-sampling's background down-weight):  0.9  -> 0.95

    Hardcoded (not read from env vars) so this trainer can be run standalone, with its own
    output folder, without touching the tunable base trainer's runs or env-var-driven configs.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.awing_weight = 20.0
        self.awing_focal_gamma = 4.0
        self.awing_soft_max_downweight = 0.95

        self.print_to_log_file(
            f"nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingHighFG: hardcoded overrides "
            f"weight={self.awing_weight}, focal_gamma={self.awing_focal_gamma}, "
            f"soft_max_downweight={self.awing_soft_max_downweight}")
