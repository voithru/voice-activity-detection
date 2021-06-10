from dataclasses import dataclass

from omegaconf import MISSING

from vad.lr_schedulers.lr_scheduler import LRScheduler


@dataclass
class RampupHoldDecayLRSchedulerConfig:
    """
    Small: 500, 5000, 40000
    base: 500, 10000, 80000
    double: 1000, 20000, 160000
    long: 1000, 20000, 320000
    """

    ramp_up_milestone: int = MISSING
    hold_milestone: int = MISSING
    decay_milestone: int = MISSING


class RampupHoldDecayLRScheduler(LRScheduler):
    """We ramp-up, hold, then exponentially decay the learning rate until it reaches 1/100 of its maximum value.

    Refer to SpecAugment paper.
    """

    config: RampupHoldDecayLRSchedulerConfig

    def __init__(self, config: RampupHoldDecayLRSchedulerConfig):
        self.config = config

    def get_factor(self, step: int) -> float:
        if step < self.config.ramp_up_milestone:
            factor = step / self.config.ramp_up_milestone
        elif step < self.config.hold_milestone:
            factor = 1
        elif step < self.config.decay_milestone:
            steps_after_decay = step - self.config.hold_milestone
            decay_total_steps = self.config.decay_milestone - self.config.hold_milestone
            factor = pow(10, steps_after_decay / decay_total_steps * -2)
        else:
            factor = 0.01
        return factor
