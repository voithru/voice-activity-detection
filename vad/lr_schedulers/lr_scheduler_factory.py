from dataclasses import dataclass
from enum import Enum
from typing import Optional

from omegaconf import MISSING
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from vad.lr_schedulers.cosine_annealing_warmup_lr_scheduler import (
    CosineAnnealingWarmupLRScheduler,
    CosineAnnealingWarmupLRSchedulerConfig,
)
from vad.lr_schedulers.lr_scheduler import ConstantLRScheduler
from vad.lr_schedulers.noam_lr_scheduler import NoamLRScheduler, NoamLRSchedulerConfig
from vad.lr_schedulers.rampup_hold_decay_lr_scheduler import (
    RampupHoldDecayLRScheduler,
    RampupHoldDecayLRSchedulerConfig,
)
from vad.lr_schedulers.warmup_linear_lr_scheduler import WarmupLinearConfig


class LRSchedulerName(Enum):
    CONSTANT = "constant"
    WARMUP_LINEAR = "warmup-linear"
    RAMPUP_HOLD_DECAY = "rampup-hold-decay"
    NOAM = "noam"
    COSINE_ANNEALING_WARMUP = "cosine-annealing-warmup"


@dataclass
class LRSchedulerConfig:
    name: str = MISSING
    warmup_linear: Optional[WarmupLinearConfig] = None
    rampup_hold_decay: Optional[RampupHoldDecayLRSchedulerConfig] = None
    noam: Optional[NoamLRSchedulerConfig] = None
    cosine_annealing_warmup: Optional[CosineAnnealingWarmupLRSchedulerConfig] = None


def create_lr_scheduler(
    optimizer: Optimizer,
    config: Optional[LRSchedulerConfig],
):
    if config is None:
        lr_factor_scheduler = ConstantLRScheduler()
        lr_scheduler = LambdaLR(optimizer, lr_factor_scheduler.get_factor)
        return lr_scheduler
    name = LRSchedulerName(config.name)
    if name == LRSchedulerName.CONSTANT:
        lr_factor_scheduler = ConstantLRScheduler()
        lr_scheduler = LambdaLR(optimizer, lr_factor_scheduler.get_factor)
    elif name == LRSchedulerName.WARMUP_LINEAR:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_linear.warmup_steps,
            num_training_steps=config.warmup_linear.total_steps,
        )
    elif name == LRSchedulerName.RAMPUP_HOLD_DECAY:
        lr_factor_scheduler = RampupHoldDecayLRScheduler(config.rampup_hold_decay)
        lr_scheduler = LambdaLR(optimizer, lr_factor_scheduler.get_factor)
    elif name == LRSchedulerName.NOAM:
        lr_factor_scheduler = NoamLRScheduler(config.noam)
        lr_scheduler = LambdaLR(optimizer, lr_factor_scheduler.get_factor)
    elif name == LRSchedulerName.COSINE_ANNEALING_WARMUP:
        lr_factor_scheduler = CosineAnnealingWarmupLRScheduler(config.cosine_annealing_warmup)
        lr_scheduler = LambdaLR(optimizer, lr_factor_scheduler.get_factor)
    else:
        raise NotImplementedError

    return lr_scheduler
