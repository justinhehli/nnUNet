from typing import List, Tuple, Union

import torch
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

from nnunetv2.inference.heatmap_export import export_single_channel_heatmap_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.data_augmentation.custom_transforms.heatmap_regression import ConvertSegToSingleChannelHeatmap
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.heatmap.nnUNetTrainerHeatmapAdaptiveWingReweighted import \
    nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling


class nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel(nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling):
    """
    Single-channel counterpart to nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling, for the
    single-label dataset case see (JunctionDetection/PreProcessing/create_nnunet_dataset_variants.py), 
    where all landmark types are merged into one foreground label

    nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling "wastes" an output channel and a loss term 
    reproducing the exact same values. This trainer instead builds a true 1-channel network and target
    (ConvertSegToSingleChannelHeatmap), and exports point predictions via 
    export_single_channel_heatmap_prediction_from_logits

    REQUIRES a one-time setup step before training (see JunctionDetection/PreProcessing/patch_single_label_plans.py): 
    the preprocessed dataset's nnUNetPlans.json must have "label_manager": "SingleHeadLabelManager" set,
    so the network is sized with 1 output channel both for training and whenever the trained model is 
    later reloaded for inference
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        assert len(self.label_manager.foreground_labels) == 1, (
            f"nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel expects a dataset with "
            f"exactly one foreground label, got {self.label_manager.foreground_labels}. Use "
            f"nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling (or one of the other multi-label "
            f"heatmap trainers) for multi-label datasets instead."
        )
        assert self.label_manager.num_segmentation_heads == 1, (
            f"nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel expects the network to be "
            f"sized with 1 output channel, got {self.label_manager.num_segmentation_heads}. Did you "
            f"run JunctionDetection/PreProcessing/patch_single_label_plans.py on this dataset's "
            f"nnUNetPlans.json before training? (It must be re-run after any re-preprocessing.)"
        )

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
        assert not is_cascaded, \
            "nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel does not support cascaded training"
        transforms = nnUNetTrainer.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm, is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        transforms.transforms.append(
            ConvertSegToSingleChannelHeatmap(sigma=self.heatmap_sigma))
        return transforms

    def get_validation_transforms(
            self,
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        assert not is_cascaded, \
            "nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel does not support cascaded training"
        transforms = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales, is_cascaded=is_cascaded, foreground_labels=foreground_labels,
            regions=regions, ignore_label=ignore_label)
        transforms.transforms.append(
            ConvertSegToSingleChannelHeatmap(sigma=self.heatmap_sigma))
        return transforms

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Duplicated from nnUNetTrainerHeatmapMSE.perform_actual_validation.
        The only difference from the multi-label version is the export call at the end of the loop:
        export_single_channel_heatmap_prediction_from_logits instead of
        export_heatmap_prediction_from_logits
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

            export_single_channel_heatmap_prediction_from_logits(
                prediction, properties, self.configuration_manager, self.plans_manager, self.dataset_json,
                join(validation_output_folder, k), save_probabilities=save_probabilities,
                threshold=self.heatmap_threshold, min_distance=self.heatmap_min_distance)

        self.print_to_log_file(
            f"Wrote validation heatmaps/point predictions to {validation_output_folder}")
