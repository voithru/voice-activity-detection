from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING
from torch.optim import Adam

try:
    from apex.optimizers import FusedAdam

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


@dataclass
class AdamConfig:
    eps: float = MISSING
    weight_decay_val: float = 0


@dataclass
class OptimizerConfig:
    name: str = MISSING
    lr: float = MISSING
    adam: Optional[AdamConfig] = None


def select_optimizer(parameters, optimizer_config: OptimizerConfig):
    if optimizer_config.name == "adam":
        optimizer = Adam(
            parameters,
            lr=optimizer_config.lr,
            betas=(0.9, 0.98),
            eps=optimizer_config.adam.eps,
            weight_decay=optimizer_config.adam.weight_decay_val,
        )

    # TODO
    # elif hparams.optimizer == "fused-adam":
    #     if APEX_AVAILABLE:
    #         optimizer = FusedAdam(
    #             parameters,
    #             lr=hparams.lr,
    #             betas=(0.9, 0.98),
    #             eps=hparams.eps,
    #             weight_decay=hparams.weight_decay_val,
    #         )
    #     else:
    #         raise ModuleNotFoundError("No module named 'apex'")
    # elif hparams.optimizer == "adam-w":
    #     optimizer = AdamW(
    #         parameters,
    #         lr=hparams.lr,
    #         betas=(0.9, 0.98),
    #         eps=hparams.eps,
    #         weight_decay=hparams.weight_decay_val,
    #     )
    # else:
    #     raise NotImplementedError

    # if hparams.swa:
    #     optimizer = SWA(
    #         optimizer=optimizer,
    #         swa_start=hparams.swa_start,
    #         swa_freq=hparams.swa_freq,
    #         swa_lr=hparams.swa_lr,
    #     )

    return optimizer
