import os
import numpy as np
import torch
import wandb

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWandb(nnUNetTrainer):
    """
    Identical to the default nnUNetTrainer, but logs to Weights & Biases.

    Requires the following environment vars to be set:
        WANDB_API_KEY           (Weights & Biases API key)
        WANDB_ENTITY            (Weights & Biases entity)
        WANDB_NNUNET_PROJECT    (Weights & Biases project name)

    Command to run:
        set -a && source ~/repos/ForkSight/Environment/.env && set +a && CUDA_VISIBLE_DEVICES=<DEVICE> nnUNetv2_train <DATASET_id> 2d <FOLD> -tr nnUNetTrainerWandb --npz
    
    Note: CUDA_VISIBLE_DEVICES=<DEVICE> makes the device with the specified index the only visible one, so nnUNet will see it as cuda:0 and log accordingly 
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.wandb_run = None
        if self.local_rank == 0:
            wandb_api_key = os.environ.get("WANDB_API_KEY", None)
            wandb_entity = os.environ.get("WANDB_ENTITY", None)
            wandb_project = os.environ.get("WANDB_NNUNET_PROJECT", None)
            if wandb_api_key is None or wandb_entity is None or wandb_project is None:
                raise RuntimeError(
                    "WANDB_API_KEY, WANDB_ENTITY, and WANDB_NNUNET_PROJECT env vars must be set for W&B logging.")

            wandb.login(key=wandb_api_key)
            self.wandb_run = wandb.init(
                entity=wandb_entity,
                project=wandb_project,
                name=(
                    f"{self.__class__.__name__}"
                    f"__{self.plans_manager.dataset_name}"
                    f"__{configuration}__fold{fold}"
                ),
                config=self._wandb_config(configuration, fold),
                reinit=True,
            )

            self.print_to_log_file(f"W&B run: {self.wandb_run.url}")

    def _wandb_config(self, configuration: str, fold: int) -> dict:
        """
        Build the W&B config dict. Override in subclasses to add
        extra fields (e.g. loss weights).
        """
        return {
            "trainer": self.__class__.__name__,
            "dataset": self.plans_manager.dataset_name,
            "configuration": configuration,
            "fold": fold,
            "initial_lr": self.initial_lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.configuration_manager.batch_size,
            "patch_size": list(self.configuration_manager.patch_size),
        }

    def on_epoch_end(self):
        super().on_epoch_end()

        if self.wandb_run is None:
            return

        log_dict = {
            "train/loss": self.logger.get_value("train_losses", step=-1),
            "validation/loss": self.logger.get_value("val_losses", step=-1),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        # try:
        #    dice_per_class = self.logger.get_value(
        #        "dice_per_class_or_region", step=-1
        #    )
        #    if dice_per_class is not None:
        #        for i, d in enumerate(dice_per_class):
        #            log_dict[f"pseudo_dice/class_{i}"] = d
        #        fg = [d for d in dice_per_class if not np.isnan(d)]
        #        if fg:
        #            log_dict["pseudo_dice/mean_fg"] = np.nanmean(fg)
        # except Exception:
        #    pass
        # try:
        #    ema = self.logger.get_value("ema_fg_dice", step=-1)
        #    if ema is not None:
        #        log_dict["ema_fg_dice"] = ema
        # except Exception:
        #    pass

        wandb.log(log_dict)

    def on_train_end(self):
        super().on_train_end()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
