from dataclasses import dataclass
from enum import Enum
from typing import Optional

from omegaconf import MISSING
from torch.optim import Adam
from transformers import AdamW

from vad.optimizers.adam import AdamConfig


class OptimizerName(Enum):
    ADAM = "adam"
    ADAM_W = "adam-w"


@dataclass
class OptimizerConfig:
    name: str = MISSING
    lr: float = MISSING
    adam: Optional[AdamConfig] = None


def create_optimizer(parameters, config: OptimizerConfig):
    name = OptimizerName(config.name)
    if name == OptimizerName.ADAM:
        optimizer = Adam(
            parameters,
            lr=config.lr,
            betas=(0.9, 0.98),
            eps=config.adam.eps,
            weight_decay=config.adam.weight_decay_val,
        )
    elif name == OptimizerName.ADAM_W:
        optimizer = AdamW(parameters, lr=config.lr, eps=config.adam.eps)
    else:
        raise NotImplementedError
    return optimizer
