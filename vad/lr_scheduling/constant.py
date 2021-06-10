from torch.optim.lr_scheduler import _LRScheduler


class Constant(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(Constant, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]
