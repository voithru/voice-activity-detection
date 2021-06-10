from dataclasses import dataclass

from omegaconf import MISSING

from vad.lr_schedulers.lr_scheduler import LRScheduler


@dataclass
class NoamLRSchedulerConfig:
    factor: int = MISSING
    d_model: int = MISSING
    warmup_steps: int = MISSING


class NoamLRScheduler(LRScheduler):
    d_model: int
    warmup_steps: int

    def __init__(self, config: NoamLRSchedulerConfig):
        self.factor = config.factor
        self.d_model = config.d_model
        self.warmup_steps = config.warmup_steps

    def get_factor(self, step: int):
        step += 1  # Avoid zero
        return (
            self.factor
            * (self.d_model ** -0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
