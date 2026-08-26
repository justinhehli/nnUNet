from nnunetv2.utilities.label_handling.label_handling import LabelManager


class SingleHeadLabelManager(LabelManager):
    """
    LabelManager that reports exactly one segmentation head, regardless of the label dict's
    background+foreground count.

    For a plain (non-region) label dict, the default LabelManager.num_segmentation_heads is
    len(all_labels) - i.e. background + foreground count. For a dataset with exactly one foreground
    label (e.g. the "combined" landmark dataset produced by
    JunctionDetection/PreProcessing/create_nnunet_dataset_variants.py), that would size the network
    with 2 output channels even though nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel's on-the-fly heatmap target
    (ConvertSegToSingleChannelHeatmap) is only 1 channel wide - the "background" and "foreground"
    channels of a heatmap-regression network aren't a softmax pair the way they are in ordinary
    segmentation; there is no "background heatmap" target anywhere in that pipeline, so a 2nd channel
    there is a wasted duplicate, not a meaningful class.

    Selected via the "label_manager" key in plans.json (see
    JunctionDetection/PreProcessing/patch_single_label_plans.py), which PlansManager.get_label_manager
    resolves identically whether called from nnUNetTrainer.initialize() (training) or
    nnUNetPredictor.initialize_from_trained_model_folder() / manual_initialization() (inference) - so
    patching plans.json once, before training, is enough to keep the network's output-channel count
    consistent everywhere without any special-casing in the trainer itself.
    """

    @property
    def num_segmentation_heads(self) -> int:
        return 1
