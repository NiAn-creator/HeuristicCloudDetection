from torch.optim.lr_scheduler import _LRScheduler, StepLR


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        lr = [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

        return lr


class check_LR(_LRScheduler):
    def __init__(self, optimizer, gamma=0.1):
        self.gamma = gamma
        self.optimizer = optimizer
        super(check_LR, self).__init__(optimizer)

    def get_lr(self):
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]