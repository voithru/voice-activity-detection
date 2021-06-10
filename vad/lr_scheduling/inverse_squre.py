from torch.optim.lr_scheduler import _LRScheduler


class InverseSquareRootSchedule(_LRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      decay_factor = args.lr * sqrt(args.warmup_updates)
      lr = decay_factor / sqrt(update_num)
    """

    def __init__(self, optimizer, max_lr, warmup_init_lr, warmup_updates, last_epoch=-1):

        if warmup_init_lr is None:
            warmup_init_lr = 0 if warmup_updates > 0 else lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (max_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = max_lr * warmup_updates ** 0.5

        self.warmup_init_lr = warmup_init_lr
        self.warmup_updates = warmup_updates

        super(InverseSquareRootSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.calculate_lr(step=self.last_epoch) for _ in self.optimizer.param_groups]

    def calculate_lr(self, step):
        if step < self.warmup_updates:
            lr = self.warmup_init_lr + step * self.lr_step
        else:
            lr = self.decay_factor * step ** -0.5
        return lr
