import os

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.cldice import SoftClDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.variants.logging.nnUNetTrainerWandb import nnUNetTrainerWandb


class DC_BCE_and_ClDice_loss(nn.Module):
    """
    Combined loss for region-based (BCE) tasks:
        w_ce * BCE + w_dice * Dice + w_cldice * clDice
    """

    def __init__(
        self,
        bce_kwargs: dict,
        soft_dice_kwargs: dict,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_cldice: float = 1.0,
        use_ignore_label: bool = False,
        skeletonize_iter: int = 15,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.weight_cldice = weight_cldice

        self.base_loss = DC_and_BCE_loss(
            bce_kwargs,
            soft_dice_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            use_ignore_label=use_ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.cldice = SoftClDiceLoss(
            skeletonize_iter=skeletonize_iter, smooth=smooth)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        base = self.base_loss(net_output, target)

        if self.weight_cldice != 0:
            # target is one-hot; use only the first foreground channel for clDice
            target_fg = target[:, :1].float()
            logits_fg = net_output[:, :1]
            cl = self.cldice(logits_fg, target_fg)
            self._last_cl = float(cl.detach())
        else:
            cl = 0
            self._last_cl = 0.0

        return base + self.weight_cldice * cl


class DC_CE_and_ClDice_loss(nn.Module):
    """
    Combined loss for multi-class (CE) tasks:
        w_ce * CE + w_dice * Dice + w_cldice * clDice
    """

    def __init__(
        self,
        ce_kwargs: dict,
        soft_dice_kwargs: dict,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_cldice: float = 1.0,
        ignore_label=None,
        skeletonize_iter: int = 10,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.weight_cldice = weight_cldice

        self.base_loss = DC_and_CE_loss(
            soft_dice_kwargs,
            ce_kwargs,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            ignore_label=ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.cldice = SoftClDiceLoss(
            skeletonize_iter=skeletonize_iter, smooth=smooth)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        base = self.base_loss(net_output, target)

        if self.weight_cldice != 0:
            # target shape: (B, 1, ...) with integer class labels
            target_fg = (target > 0).float()
            logits_fg = net_output[:, 1:2]  # foreground logit (class index 1)
            cl = self.cldice(logits_fg, target_fg)
            self._last_cl = float(cl.detach())
        else:
            cl = 0
            self._last_cl = 0.0

        return base + self.weight_cldice * cl


class nnUNetTrainerClDiceLoss(nnUNetTrainerWandb):
    """
    nnUNetTrainer that logs to Weights & Biases and uses:
        w_ce * (BCE or CE) + w_dice * Dice + w_cldice * clDice

    Automatically selects BCE (region-based tasks) or CE (multi-class tasks)
    the same way nnUNetTrainer does.

    Loss weights are read from environment variables:
        NNUNET_CE_WEIGHT       weight for BCE/CE   (default 1.0)
        NNUNET_DICE_WEIGHT     weight for Dice      (default 1.0)
        NNUNET_CLDICE_WEIGHT   weight for clDice    (default 1.0)
        NNUNET_CLDICE_ITER     skeletonise iters    (default 15)
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.w_ce = float(os.environ.get("NNUNET_CE_WEIGHT", 1.0))
        self.w_dice = float(os.environ.get("NNUNET_DICE_WEIGHT", 1.0))
        self.w_cldice = float(os.environ.get("NNUNET_CLDICE_WEIGHT", 1.0))
        self.cldice_iter = int(os.environ.get("NNUNET_CLDICE_ITER", 15))

        self._update_wandb_config()

        self.print_to_log_file(
            f"Loss weights: CE: {self.w_ce}, Dice: {self.w_dice}, clDice: {self.w_cldice} "
            f"(skeletonize_iter={self.cldice_iter})"
        )

    def _update_wandb_config(self):
        self.wandb_run.config.update({
            "CE_weight": self.w_ce,
            "Dice_weight": self.w_dice,
            "clDice_weight": self.w_cldice,
            "clDice_iter": self.cldice_iter,
        })

    def _get_loss_module(self):
        return self.loss.loss if isinstance(self.loss, DeepSupervisionWrapper) else self.loss

    def train_step(self, batch: dict) -> dict:
        ret = super().train_step(batch)
        ret['cldice_loss'] = self._get_loss_module()._last_cl
        return ret

    def on_train_epoch_end(self, train_outputs: list) -> None:
        super().on_train_epoch_end(train_outputs)
        mean_cl = float(np.mean([o['cldice_loss'] for o in train_outputs]))
        self.print_to_log_file(
            f'cldice_loss (unweighted): {np.round(mean_cl, decimals=4)}')

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_BCE_and_ClDice_loss(
                bce_kwargs={},
                soft_dice_kwargs={
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp,
                },
                weight_ce=self.w_ce,
                weight_dice=self.w_dice,
                weight_cldice=self.w_cldice,
                use_ignore_label=self.label_manager.ignore_label is not None,
                skeletonize_iter=self.cldice_iter,
            )
        else:
            loss = DC_CE_and_ClDice_loss(
                ce_kwargs={},
                soft_dice_kwargs={
                    'batch_dice': self.configuration_manager.batch_dice,
                    'smooth': 1e-5,
                    'do_bg': False,
                    'ddp': self.is_ddp,
                },
                weight_ce=self.w_ce,
                weight_dice=self.w_dice,
                weight_cldice=self.w_cldice,
                ignore_label=self.label_manager.ignore_label,
                skeletonize_iter=self.cldice_iter,
            )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i)
                               for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
