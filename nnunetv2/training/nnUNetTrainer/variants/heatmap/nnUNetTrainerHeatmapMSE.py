import os
from typing import List, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

from nnunetv2.inference.heatmap_export import export_heatmap_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.data_augmentation.custom_transforms.heatmap_regression import ConvertSegToMultiChannelHeatmap
from nnunetv2.training.loss.heatmap_mse import SigmoidMSELoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.logging.nnUNetTrainerWandb import nnUNetTrainerWandb


class nnUNetTrainerHeatmapMSE(nnUNetTrainerWandb):
    """
    Multi-instance, multi-channel Gaussian-heatmap regression, trained with MSE 
    (see nnunetv2.training.loss.heatmap_mse.SigmoidMSELoss)

    On-disk labels stay a discrete "NxN pixel blob at each landmark's coordinate" segmentation map 
    (see JunctionDetection/PreProcessing/create_nnunet_heatmap_dataset.py), so nnU-Net's
    default cropping/resampling/foreground-oversampling all keep working unmodified. The only custom
    pieces are:
      - ConvertSegToMultiChannelHeatmap, appended as the last training/validation transform, which
        turns the (already spatially-augmented) blob map into a dense (num_classes + 1, H, W) float
        heatmap target: channel 0 is the "combined" heatmap (any landmark), channels 1..N are
        per-class. Connected components (and therefore instances) are found independently per class,
      - SigmoidMSELoss as the loss.
      - perform_actual_validation, which replaces nnU-Net's default argmax/Dice-based validation
        (meaningless for a regression target) with heatmap + point-prediction export.

    reads the following env vars:
        WANDB_NNUNET_LANDMARK_PROJECT Weights & Biases project to be logged to
        NNUNET_HEATMAP_SIGMA          gaussian sigma in pixels for the heatmap targets (default 3.0)
        NNUNET_HEATMAP_THRESHOLD      confidence threshold for peak extraction at validation/inference (default 0.5)
        NNUNET_HEATMAP_MIN_DISTANCE   minimum pixel distance between two detected peaks (default 3)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device,
                         override_wandb_project=os.environ.get("WANDB_NNUNET_LANDMARK_PROJECT", None))

        self.enable_deep_supervision = False
        self.heatmap_sigma = float(os.environ.get("NNUNET_HEATMAP_SIGMA", 3.0))
        self.heatmap_threshold = float(
            os.environ.get("NNUNET_HEATMAP_THRESHOLD", 0.5))
        self.heatmap_min_distance = int(
            os.environ.get("NNUNET_HEATMAP_MIN_DISTANCE", 3))

        self.print_to_log_file(
            f"nnUNetTrainerHeatmapMSE: sigma={self.heatmap_sigma}, threshold={self.heatmap_threshold}, "
            f"min_distance={self.heatmap_min_distance}, deep_supervision={self.enable_deep_supervision}"
        )

    def _build_loss(self):
        return SigmoidMSELoss()

    def get_training_transforms(
            self, patch_size: Union[torch.Tensor, Tuple[int, ...]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        assert not is_cascaded, "nnUNetTrainerHeatmapMSE does not support cascaded training"
        transforms = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        transforms.transforms.append(
            ConvertSegToMultiChannelHeatmap(len(self.label_manager.foreground_labels), sigma=self.heatmap_sigma))
        return transforms

    def get_validation_transforms(
            self,
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        assert not is_cascaded, "nnUNetTrainerHeatmapMSE does not support cascaded training"
        transforms = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales, is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        transforms.transforms.append(
            ConvertSegToMultiChannelHeatmap(len(self.label_manager.foreground_labels), sigma=self.heatmap_sigma))
        return transforms

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Replaces nnU-Net's default argmax/Dice-based validation (meaningless here - there's no
        discrete segmentation to argmax into) with heatmap + point-prediction export for each
        validation case, using the same export_heatmap_prediction_from_logits used at real inference
        time. Simplified relative to nnU-Net's own perform_actual_validation: single
        process, no DDP or cascade support (this trainer is only used for the 2D, single-GPU-per-fold
        junction-detection task).
        """
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        validation_output_folder = join(self.output_folder, 'validation')
        maybe_mkdir_p(validation_output_folder)

        _, val_keys = self.do_split()
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

        for k in dataset_val.identifiers:
            self.print_to_log_file(f"predicting {k}")
            data, _, _, properties = dataset_val.load_case(k)
            data = torch.from_numpy(data[:])

            with torch.no_grad():
                prediction = predictor.predict_sliding_window_return_logits(
                    data).cpu().numpy()

            export_heatmap_prediction_from_logits(
                prediction, properties, self.configuration_manager, self.plans_manager, self.dataset_json,
                join(validation_output_folder, k), save_probabilities=save_probabilities,
                threshold=self.heatmap_threshold, min_distance=self.heatmap_min_distance)

        self.print_to_log_file(
            f"Wrote validation heatmaps/point predictions to {validation_output_folder}")
