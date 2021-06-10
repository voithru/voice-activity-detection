from dataclasses import dataclass

from omegaconf import MISSING

from vad.lr_schedulers.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from vad.lr_schedulers.lr_scheduler import LRScheduler


@dataclass
class CosineAnnealingWarmupLRSchedulerConfig:
    initial_cycle_steps: int = MISSING
    cycle_steps_multiplier: float = MISSING
    peak: float = MISSING
    trough: float = MISSING
    warmup_steps: int = MISSING
    shrink_multiplier: float = MISSING


class CosineAnnealingWarmupLRScheduler(LRScheduler):
    def __init__(self, config: CosineAnnealingWarmupLRSchedulerConfig):
        self.cosine_annealing_warmup = CosineAnnealingWarmupRestarts(
            config.initial_cycle_steps,
            config.cycle_steps_multiplier,
            config.peak,
            config.trough,
            config.warmup_steps,
            config.shrink_multiplier,
        )

    def get_factor(self, step: int) -> float:
        return self.cosine_annealing_warmup.get(step)
