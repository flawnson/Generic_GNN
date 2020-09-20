""" Implement Lookahead optimizer, Cosine anenaling optimizer, etc."""
import torch
import numpy as np

from torch.optim.optimizer import Optimizer
from utils.logger import log


class _LRScheduler(object):
    """ Heavily inspired by PyTorch's learning rate scheduler code, used here for testing purposes """
    def __init__(self, optimizer, last_epoch=None):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        else:
            self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if not last_epoch:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   f"in param_groups[{i}] when resuming an optimizer")
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch


class CosineAnnealingWarmRestartsOptim:
    """Derived from https://arxiv.org/pdf/1608.03983.pdf eqn. 5, work in progress"""
    def __init__(self, max_lr, min_lr, epochs_since_last_restart, current_epoch):
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self):
        self.eq = self.min_lr + (self.max_lr - self.min_lr) / 2 * (1 + np.cos(np.array([self.since_epoch / self.cur_epoch * np.pi])))

    def __call__(self, *args, **kwargs):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts()


class OptimizerObj(object):
    def __init__(self, config, params):
        self.optim_config = config["optim_config"]
        self.params = params

        if config["optim"].casefold() == "adam":
            self.optim_obj = torch.optim.Adam(params, **self.optim_config["kwargs"])
        elif config["optim"].casefold() == "sgd":
            self.optim_obj = torch.optim.SGD(params, **self.optim_config["kwargs"])
        elif config["optim"].casefold() == "adagrad":
            self.optim_obj = torch.optim.Adagrad(params, **self.optim_config["kwargs"])
        elif config["optim"].casefold() == "rmsprop":
            self.optim_obj = torch.optim.RMSprop(params, **self.optim_config["kwargs"])
        elif config["optim"].casefold() == "adadelta":
            self.optim_obj = torch.optim.Adadelta(params, **self.optim_config["kwargs"])
        else:
            log.info(f"Optimizer {self.config['optim']} not understood")
            raise NotImplementedError(f"Optimizer {self.config['optim']} not implemented")


class LRScheduler(object):
    def __init__(self, config, optim_obj):
        self.optim_config = config["optim_config"]
        if self.optim_config["scheduler"].casefold() == "cawr":
            self.schedule_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_obj, **self.optim_config["scheduler_kwargs"])
        elif self.optim_config["scheduler"].casefold() == "multistep":
            self.schedule_obj = torch.optim.lr_scheduler.MultiStepLR(optim_obj, **self.optim_config["scheduler_kwargs"])
        elif self.optim_config["scheduler"].casefold() == "cyclic":
            self.schedule_obj = torch.optim.lr_scheduler.CyclicLR(optim_obj, **self.optim_config["scheduler_kwargs"])
        else:
            self.schedule_obj = None
            log.info(f"Scheduler {self.config['scheduler']} not provided or not understood")
