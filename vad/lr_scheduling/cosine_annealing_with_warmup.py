import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer,
        update_steps,
        step_multiple=1,
        max_annealing_lr=0.1,
        annealing_warmup_steps=0,
        gamma=1.0,
        last_epoch=-1,
    ):
        if update_steps <= 0 or not isinstance(update_steps, int):
            raise ValueError(
                "Expected positive integer update_steps, but got {}".format(update_steps)
            )
        if step_multiple < 1 or not isinstance(step_multiple, int):
            raise ValueError(
                "Expected integer step_multiple >= 1, but got {}".format(step_multiple)
            )
        if annealing_warmup_steps < 0 or not isinstance(annealing_warmup_steps, int):
            raise ValueError(
                "Expected positive integer annealing_warmup_steps, but got {}".format(
                    annealing_warmup_steps
                )
            )
        self.init_update_steps = update_steps
        self.step_multiple = step_multiple
        self.base_max_annealing_lr = max_annealing_lr
        self.max_annealing_lr = max_annealing_lr
        self.annealing_warmup_steps = annealing_warmup_steps
        self.update_steps_i = update_steps
        self.gamma = gamma
        self.cycle = 0
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch < self.annealing_warmup_steps:
            return [
                (self.max_annealing_lr - base_lr) * self.last_epoch / self.annealing_warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_annealing_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.annealing_warmup_steps)
                        / (self.update_steps_i - self.annealing_warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            if self.last_epoch >= self.update_steps_i:
                self.cycle += 1
                self.last_epoch = self.last_epoch - self.update_steps_i
                self.update_steps_i = (
                    self.update_steps_i - self.annealing_warmup_steps
                ) * self.step_multiple + self.annealing_warmup_steps
        else:
            if epoch >= self.init_update_steps:
                if self.step_multiple == 1:
                    self.last_epoch = epoch % self.init_update_steps
                    self.cycle = epoch // self.init_update_steps
                else:
                    n = int(
                        math.log(
                            (epoch / self.init_update_steps * (self.step_multiple - 1) + 1),
                            self.step_multiple,
                        )
                    )
                    self.cycle = n
                    self.last_epoch = epoch - self.init_update_steps * (
                        self.step_multiple ** n - 1
                    ) / (self.step_multiple - 1)
                    self.update_steps_i = self.init_update_steps * self.step_multiple ** (n)
            else:
                self.update_steps_i = self.init_update_steps

        self.max_annealing_lr = self.base_max_annealing_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
