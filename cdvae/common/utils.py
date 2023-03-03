import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

STATS_KEY: str = "stats"


# Adapted from https://github.com/hobogalaxy/lightning-hydra-template/blob/6bf03035107e12568e3e576e82f83da0f91d6a11/src/utils/template_utils.py#L125
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel()
                                               for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # (this is just a trick to prevent trainer from logging hparams of model, since we already did that above)
    trainer.logger.log_hyperparams = lambda params: None

PROJECT_ROOT: Path = Path(__file__).parents[2]
assert (
    PROJECT_ROOT.exists()
)
