from torch.optim.lr_scheduler import _LRScheduler

"""
if type == "small":
    self.milestones = [500, 5000, 40000]
elif type == "base":
    self.milestones = [500, 10000, 80000]
elif type == "double":
    self.milestones = [1000, 20000, 160000]
elif type == "long":
    self.milestones = [1000, 20000, 320000]
else:
"""


class RampUpHoldDecay(_LRScheduler):
    """We ramp-up, hold, then exponentially decay the learning rate until it reaches 1/100 of its maximum value.

    Refer to SpecAugment paper.
    """

    def __init__(
        self,
        optimizer,
        high_plateau_lr,
        ramp_up_milestone,
        hold_milestone,
        decay_milestone,
        last_epoch=-1,
    ):
        self.high_plateau_lr = high_plateau_lr
        self.ramp_up_milestone = ramp_up_milestone
        self.hold_milestone = hold_milestone
        self.decay_milestone = decay_milestone

        super(RampUpHoldDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.calculate_lr(step=self.last_epoch) for base_lr in self.base_lrs]

    def calculate_lr(self, step):
        if step < self.ramp_up_milestone:
            lr = self.high_plateau_lr * (step / self.ramp_up_milestone)
        elif step < self.hold_milestone:
            lr = self.high_plateau_lr
        elif step < self.decay_milestone:
            steps_after_decay = step - self.hold_milestone
            decay_total_steps = self.decay_milestone - self.hold_milestone
            lr = self.high_plateau_lr * pow(10, steps_after_decay / decay_total_steps * -2)
        else:
            lr = self.high_plateau_lr * 0.01
        return lr
