from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class CosineAnnealingConfig:
    update_steps: int = MISSING
    step_multiple: int = 1
    max_lr: float = 0.1
    warmup_steps: int = 0
    gamma: float = 1.0


@dataclass
class InverseSquareConfig:
    warmup_init_lr: float = MISSING
    warmup_steps: int = MISSING


@dataclass
class NoamConfig:
    factor: float = MISSING
    d_model: int = MISSING
    warmup_steps: int = MISSING


@dataclass
class RampUpHoldDecayConfig:
    high_plateau_lr: float = MISSING
    ramp_up_milestone: int = MISSING
    hold_milestone: int = MISSING
    decay_milestone: int = MISSING


@dataclass
class CyclicLRConfig:
    max_lr: float = MISSING
    step_size_up: int = MISSING
    step_size_down: int = MISSING
    mode: str = MISSING
    gamma: float = MISSING


@dataclass
class ExponentialConfig:
    gamma: float = MISSING


@dataclass
class ReduceLROnPlateauConfig:
    factor: float = MISSING
    patience: int = MISSING
    threshold: float = MISSING
    threshold_mode: str = MISSING


@dataclass
class LRSchedulerConfig:
    name: str = MISSING
    ramp_up_hold_decay: Optional[RampUpHoldDecayConfig] = None
    noam: Optional[NoamConfig] = None
    cyclic: Optional[CyclicLRConfig] = None
    cosine_annealing: Optional[CosineAnnealingConfig] = None
    exponential: Optional[ExponentialConfig] = None
    inverse_squre: Optional[InverseSquareConfig] = None
    reduce_on_plateau: Optional[ReduceLROnPlateauConfig] = None
