import copy
import os

import torch
from torch import autocast

from nnunetv2.training.loss.adaptive_wing import ReweightedAdaptiveWingLoss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.variants.heatmap.nnUNetTrainerHeatmapAdaptiveWing import \
    nnUNetTrainerHeatmapAdaptiveWing
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerHeatmapAdaptiveWingFocal(nnUNetTrainerHeatmapAdaptiveWing):
    """
    nnUNetTrainerHeatmapAdaptiveWing with focal-style hard-positive upweighting: at GT-positive
    pixels, the AWing loss is multiplied by (1 - pred)^gamma (pred detached), so pixels the network
    is currently confidently wrong about (raw output near 0 at a true landmark - "confident false
    negatives") count for relatively more of the loss than already-easy positives. See
    ReweightedAdaptiveWingLoss for the exact reweighting.

    reads the following additional env var (on top of nnUNetTrainerHeatmapAdaptiveWing's):
        NNUNET_AWING_FOCAL_GAMMA  focal modulating exponent  (default 2.0)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.awing_focal_gamma = float(
            os.environ.get("NNUNET_AWING_FOCAL_GAMMA", 2.0))

        self.print_to_log_file(
            f"nnUNetTrainerHeatmapAdaptiveWingFocal: gamma={self.awing_focal_gamma}")

    def _build_loss(self):
        return ReweightedAdaptiveWingLoss(
            omega=self.awing_omega, theta=self.awing_theta, epsilon=self.awing_epsilon,
            alpha=self.awing_alpha, gamma=self.awing_focal_gamma, max_downweight=None,
            use_weight_map=self.awing_use_weight_map, weight=self.awing_weight,
            dilation_threshold=self.awing_dilation_threshold)


class nnUNetTrainerHeatmapAdaptiveWingSoftSampling(nnUNetTrainerHeatmapAdaptiveWing):
    """
    nnUNetTrainerHeatmapAdaptiveWing with soft-sampling background down-weighting: a frozen snapshot
    of the model, refreshed every N epochs (not every step - refreshing every step would let the
    model's own current, possibly-wrong confidence erase its own training signal in a feedback loop),
    is used to scale down the AWing loss at GT-background pixels the snapshot is itself confident
    about, capped at a configurable max down-weight. See ReweightedAdaptiveWingLoss for the exact
    reweighting.

    Overrides train_step/validation_step (not just _build_loss) because the loss needs the frozen
    snapshot's prediction on the same input image, which nnUNetTrainer's default train_step/
    validation_step don't expose to self.loss (they only pass it (output, target)).

    reads the following additional env vars (on top of nnUNetTrainerHeatmapAdaptiveWing's):
        NNUNET_AWING_SOFT_REFRESH_EVERY_N_EPOCHS  epochs between snapshot refreshes  (default 10)
        NNUNET_AWING_SOFT_MAX_DOWNWEIGHT          cap on the down-weight, in [0, 1]  (default 0.9)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.awing_soft_refresh_every_n_epochs = int(
            os.environ.get("NNUNET_AWING_SOFT_REFRESH_EVERY_N_EPOCHS", 10))
        self.awing_soft_max_downweight = float(
            os.environ.get("NNUNET_AWING_SOFT_MAX_DOWNWEIGHT", 0.9))
        self._snapshot_network = None

        self.print_to_log_file(
            f"nnUNetTrainerHeatmapAdaptiveWingSoftSampling: "
            f"refresh_every_n_epochs={self.awing_soft_refresh_every_n_epochs}, "
            f"max_downweight={self.awing_soft_max_downweight}")

    def _build_loss(self):
        return ReweightedAdaptiveWingLoss(
            omega=self.awing_omega, theta=self.awing_theta, epsilon=self.awing_epsilon,
            alpha=self.awing_alpha, gamma=None, max_downweight=self.awing_soft_max_downweight,
            use_weight_map=self.awing_use_weight_map, weight=self.awing_weight,
            dilation_threshold=self.awing_dilation_threshold)

    def _refresh_snapshot_if_due(self):
        if self._snapshot_network is not None and \
                self.current_epoch % self.awing_soft_refresh_every_n_epochs != 0:
            return

        mod = self.network
        if isinstance(mod, torch.nn.parallel.DistributedDataParallel):
            mod = mod.module
        if hasattr(mod, "_orig_mod"):  # unwrap torch.compile's OptimizedModule, same as
            mod = mod._orig_mod       # nnUNetTrainer.set_deep_supervision_enabled does

        self._snapshot_network = copy.deepcopy(mod)
        self._snapshot_network.eval()
        for p in self._snapshot_network.parameters():
            p.requires_grad_(False)
        self.print_to_log_file(
            f"nnUNetTrainerHeatmapAdaptiveWingSoftSampling: refreshed snapshot at epoch {self.current_epoch}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._refresh_snapshot_if_due()

    def _snapshot_predict(self, data: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._snapshot_network(data)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            snapshot_output = self._snapshot_predict(data)
            l = self.loss(output, target, snapshot_output)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            snapshot_output = self._snapshot_predict(data)
            del data
            l = self.loss(output, target, snapshot_output)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (
                torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


class nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling(nnUNetTrainerHeatmapAdaptiveWingSoftSampling):
    """
    Combination of nnUNetTrainerHeatmapAdaptiveWingFocal and
    nnUNetTrainerHeatmapAdaptiveWingSoftSampling: focal-style hard-positive upweighting AND
    snapshot-based soft-sampling background down-weighting, multiplied together. See
    ReweightedAdaptiveWingLoss for the exact reweighting.

    reads the same additional env vars as both parents:
        NNUNET_AWING_FOCAL_GAMMA                  focal modulating exponent          (default 2.0)
        NNUNET_AWING_SOFT_REFRESH_EVERY_N_EPOCHS  epochs between snapshot refreshes  (default 10)
        NNUNET_AWING_SOFT_MAX_DOWNWEIGHT          cap on the down-weight, in [0, 1]  (default 0.9)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.awing_focal_gamma = float(
            os.environ.get("NNUNET_AWING_FOCAL_GAMMA", 2.0))

        self.print_to_log_file(
            f"nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling: gamma={self.awing_focal_gamma}")

    def _build_loss(self):
        return ReweightedAdaptiveWingLoss(
            omega=self.awing_omega, theta=self.awing_theta, epsilon=self.awing_epsilon,
            alpha=self.awing_alpha, gamma=self.awing_focal_gamma,
            max_downweight=self.awing_soft_max_downweight,
            use_weight_map=self.awing_use_weight_map, weight=self.awing_weight,
            dilation_threshold=self.awing_dilation_threshold)
