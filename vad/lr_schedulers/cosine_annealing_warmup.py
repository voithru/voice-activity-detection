# Refer to https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup


import math
from typing import Tuple


class CosineAnnealingWarmupRestarts:
    initial_cycle_steps: int  # T_0
    cycle_steps_multiplier: float  # T_mult
    peak: float  # base_eta_max
    trough: float  # base_lr
    warmup_steps: int  # T_up
    shrink_multiplier: float  # gamma

    def __init__(
        self,
        initial_cycle_steps: int,
        cycle_steps_multiplier: float,
        peak: float,
        trough: float,
        warmup_steps: int,
        shrink_multiplier: float,
    ):
        self.initial_cycle_steps = initial_cycle_steps
        self.cycle_steps_multiplier = cycle_steps_multiplier
        self.peak = peak
        self.trough = trough
        self.warmup_steps = warmup_steps
        self.shrink_multiplier = shrink_multiplier

    def get(self, step: int) -> float:
        current_cycle, current_step, current_cycle_steps = self.get_current(step)
        # current_cycle: int  # cycle
        # current_step: int  # T_cur
        # current_cycle_steps: int  # T_i

        current_peak = self.peak * (self.shrink_multiplier ** current_cycle)

        if current_step == -1:
            return self.trough
        elif current_step < self.warmup_steps:
            return (current_peak - self.trough) * current_step / self.warmup_steps + self.trough
        else:
            change = (current_step - self.warmup_steps) / (current_cycle_steps - self.warmup_steps)
            return self.trough + (current_peak - self.trough) * (1 + math.cos(math.pi * change)) / 2

    def get_current(self, step: int) -> Tuple[int, int, int]:
        if step >= self.initial_cycle_steps:
            if self.cycle_steps_multiplier == 1:
                current_cycle = step // self.initial_cycle_steps
                current_step = step % self.initial_cycle_steps
                current_cycle_steps = self.initial_cycle_steps
            else:
                current_cycle = int(
                    math.log(
                        (step / self.initial_cycle_steps * (self.cycle_steps_multiplier - 1) + 1),
                        self.cycle_steps_multiplier,
                    )
                )
                current_step = step - self.initial_cycle_steps * (
                    self.cycle_steps_multiplier ** current_cycle - 1
                ) // (self.cycle_steps_multiplier - 1)
                current_cycle_steps = (
                    self.initial_cycle_steps * self.cycle_steps_multiplier ** current_cycle
                )
        else:
            current_cycle = 0
            current_step = step
            current_cycle_steps = self.initial_cycle_steps

        return current_cycle, current_step, current_cycle_steps
