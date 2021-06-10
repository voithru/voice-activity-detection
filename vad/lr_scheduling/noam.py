from torch.optim.lr_scheduler import _LRScheduler


class Noam(_LRScheduler):
    """We ramp-up, hold, then exponentially decay the learning rate until it reaches 1/100 of its maximum value.

    Refer to SpecAugment paper.
    """

    def __init__(self, optimizer, factor, d_model, warmup_steps, last_epoch=-1):
        self.factor = factor
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        super(Noam, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.calculate_lr(step=self.last_epoch) for base_lr in self.base_lrs]

    def calculate_lr(self, step):
        step = step + 1  # Avoid 0 step
        lr = (
            self.factor
            * self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        return lr
