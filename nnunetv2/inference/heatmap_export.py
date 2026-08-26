from typing import Dict, List, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from skimage.feature import peak_local_max

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def revert_heatmap_to_original_geometry(predicted_logits: Union[torch.Tensor, np.ndarray],
                                        plans_manager: PlansManager,
                                        configuration_manager: ConfigurationManager,
                                        properties_dict: dict,
                                        num_threads_torch: int = default_num_processes) -> np.ndarray:
    """
    Sigmoid-activates multi-channel heatmap logits and reverts resampling/cropping/transpose back
    to the original image's pixel space.

    Mirrors nnunetv2.inference.export_prediction.convert_predicted_logits_to_segmentation_with_correct_shape,
    but keeps the full float probability map instead of collapsing it via argmax (there is no
    "segmentation" here, only per-channel heatmaps).
    """
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if not isinstance(predicted_logits, torch.Tensor):
        predicted_logits = torch.from_numpy(predicted_logits)
    probabilities = torch.sigmoid(predicted_logits.float())

    # resample to original (pre-resampling) shape, same logic as export_prediction.py
    spacing_transposed = [properties_dict['spacing'][i]
                          for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    probabilities = configuration_manager.resampling_fn_probabilities(
        probabilities, properties_dict['shape_after_cropping_and_before_resampling'], current_spacing,
        [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])

    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.from_numpy(probabilities)

    # Revert cropping with a plain zero-fill - deliberately NOT LabelManager.revert_cropping_on_probabilities,
    # which (for non-region tasks) sets channel 0 to 1 outside the crop bbox
    probabilities_reverted_cropping = torch.zeros(
        (probabilities.shape[0], *properties_dict['shape_before_cropping']), dtype=torch.float32)
    probabilities_reverted_cropping = insert_crop_into_image(
        probabilities_reverted_cropping, probabilities, properties_dict['bbox_used_for_cropping'])
    del probabilities

    if isinstance(probabilities_reverted_cropping, torch.Tensor):
        probabilities_reverted_cropping = probabilities_reverted_cropping.cpu().numpy()

    # revert transpose
    probabilities_reverted_cropping = probabilities_reverted_cropping.transpose(
        [0] + [i + 1 for i in plans_manager.transpose_backward])

    torch.set_num_threads(old_threads)
    return probabilities_reverted_cropping


def extract_points_from_heatmap(probabilities: np.ndarray, threshold: float = 0.5, min_distance: int = 3,
                                include_combined_channel: bool = False) -> Dict[int, List[tuple]]:
    """
    probabilities: (C, 1, H, W) array in [0, 1]. Channel 0 is the "combined" heatmap, channels 1..N are
    per-class heatmaps (see ConvertSegToMultiChannelHeatmap). For each per-class channel, finds local
    maxima above `threshold` at least `min_distance` apart, so any number (zero, one, or many) of
    same-class landmarks can be recovered 
    Returns {channel_idx: [(x, y, score), ...]}.
    """
    start_channel = 0 if include_combined_channel else 1
    points_by_channel: Dict[int, List[tuple]] = {}
    for c in range(start_channel, probabilities.shape[0]):
        channel_map = np.squeeze(probabilities[c])
        coords = peak_local_max(
            channel_map, min_distance=min_distance, threshold_abs=threshold)
        points_by_channel[c] = [(int(x), int(y), float(
            channel_map[y, x])) for y, x in coords]
    return points_by_channel


def export_heatmap_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor],
                                          properties_dict: dict,
                                          configuration_manager: ConfigurationManager,
                                          plans_manager: PlansManager,
                                          dataset_json_dict_or_file: Union[dict, str],
                                          output_file_truncated: str,
                                          save_probabilities: bool = False,
                                          threshold: float = 0.5,
                                          min_distance: int = 3,
                                          num_threads_torch: int = default_num_processes) -> None:
    """
    Same (except for additional optional args) call signature as nnunetv2.inference.export_prediction.export_prediction_from_logits 
    (so it can be dropped into the same predict/export loops), but for heatmap-regression predictions: writes
    `output_file_truncated + '.json'`, a per-label list of detected point coordinates (in original
    image space) with confidence scores, and optionally `output_file_truncated + '.npz'` with the full
    reverted probability map.
    """
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    probabilities = revert_heatmap_to_original_geometry(
        predicted_array_or_file, plans_manager, configuration_manager, properties_dict,
        num_threads_torch=num_threads_torch)
    del predicted_array_or_file

    if save_probabilities:
        np.savez_compressed(output_file_truncated + '.npz',
                            probabilities=probabilities.astype(np.float16))

    points_by_channel = extract_points_from_heatmap(
        probabilities, threshold=threshold, min_distance=min_distance)
    label_to_name = {v: k for k,
                     v in dataset_json_dict_or_file['labels'].items()}
    points_by_label = {
        label_to_name.get(channel, str(channel)): [
            {'x': x, 'y': y, 'score': score} for x, y, score in points
        ]
        for channel, points in points_by_channel.items()
    }
    save_json(points_by_label, output_file_truncated + '.json')


def export_single_channel_heatmap_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor],
                                                         properties_dict: dict,
                                                         configuration_manager: ConfigurationManager,
                                                         plans_manager: PlansManager,
                                                         dataset_json_dict_or_file: Union[dict, str],
                                                         output_file_truncated: str,
                                                         save_probabilities: bool = False,
                                                         threshold: float = 0.5,
                                                         min_distance: int = 3,
                                                         num_threads_torch: int = default_num_processes) -> None:
    """
    Single-channel counterpart to export_heatmap_prediction_from_logits, for models trained with
    nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel on a dataset with exactly one
    foreground label (see    JunctionDetection/PreProcessing/create_nnunet_dataset_variants.py) -
    the predicted array has just 1 channel (the one landmark class)

    Same call signature (except for additional optional args) as
    nnunetv2.inference.export_prediction.export_prediction_from_logits, and writes the same
    `output_file_truncated + '.json'` / (optional) `.npz` outputs as
    export_heatmap_prediction_from_logits.
    """
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    probabilities = revert_heatmap_to_original_geometry(
        predicted_array_or_file, plans_manager, configuration_manager, properties_dict,
        num_threads_torch=num_threads_torch)
    del predicted_array_or_file

    assert probabilities.shape[0] == 1, \
        f"export_single_channel_heatmap_prediction_from_logits expects a single-channel heatmap, " \
        f"got {probabilities.shape[0]} channels"

    if save_probabilities:
        np.savez_compressed(output_file_truncated + '.npz',
                            probabilities=probabilities.astype(np.float16))

    points_by_channel = extract_points_from_heatmap(
        probabilities, threshold=threshold, min_distance=min_distance, include_combined_channel=True)

    foreground_labels = [name for name, label_id in dataset_json_dict_or_file['labels'].items()
                         if name != 'background' and name != 'ignore']
    assert len(foreground_labels) == 1, \
        f"export_single_channel_heatmap_prediction_from_logits expects exactly one foreground label " \
        f"in dataset.json, got {foreground_labels}"
    label_name = foreground_labels[0]

    points_by_label = {
        label_name: [{'x': x, 'y': y, 'score': score}
                     for x, y, score in points_by_channel[0]]
    }
    save_json(points_by_label, output_file_truncated + '.json')
