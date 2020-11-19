""" Implement Lookahead optimizer, Cosine anenaling optimizer, etc."""
import torch
import weakref
import warnings
import numpy as np

from torch.optim.optimizer import Optimizer
from functools import wraps
from utils.logger import log


class _LRScheduler(object):
    """Need to implement and test (taken from PyTorch lr_scheduler.py in optim lib"""
    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingWarmRestartsOptim:
    """Derived from https://arxiv.org/pdf/1608.03983.pdf eqn. 5, work in progress"""
    def __init__(self, max_lr, min_lr, epochs_since_last_restart, current_epoch):
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_lr(self):
        self.eq = self.min_lr + (self.max_lr - self.min_lr) / 2 * (1 + np.cos(np.array([self.since_epoch / self.cur_epoch * np.pi])))

    def __call__(self, *args, **kwargs):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts()


class OptimizerObj(Optimizer):
    def __init__(self, params, config):
        super(OptimizerObj, self).__init__(params, config)
        self.optim_config = config["optim_config"]
        self.param_groups = None
        self.params = params

        if self.optim_config["optim"].casefold() == "adam":
            self.optim_obj = torch.optim.Adam(params, **self.optim_config["optim_kwargs"])
        elif self.optim_config["optim"].casefold() == "sgd":
            self.optim_obj = torch.optim.SGD(params, **self.optim_config["optim_kwargs"])
        elif self.optim_config["optim"].casefold() == "adagrad":
            self.optim_obj = torch.optim.Adagrad(params, **self.optim_config["optim_kwargs"])
        elif self.optim_config["optim"].casefold() == "rmsprop":
            self.optim_obj = torch.optim.RMSprop(params, **self.optim_config["optim_kwargs"])
        elif self.optim_config["optim"].casefold() == "adadelta":
            self.optim_obj = torch.optim.Adadelta(params, **self.optim_config["optim_kwargs"])
        else:
            log.info(f"Optimizer {self.optim_config['optim']} not understood")
            raise NotImplementedError(f"Optimizer {self.optim_config['optim']} not implemented")


class LRScheduler():
    def __init__(self, config, optim_obj):
        # super(LRScheduler, self).__init__(optim_obj)
        self.optim_config = config["optim_config"]
        try:
            if self.optim_config["scheduler"].casefold() == "cawr":
                self.schedule_obj = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_obj, **self.optim_config["scheduler_kwargs"])
            elif self.optim_config["scheduler"].casefold() == "multistep":
                self.schedule_obj = torch.optim.lr_scheduler.MultiStepLR(optim_obj, **self.optim_config["scheduler_kwargs"])
            elif self.optim_config["scheduler"].casefold() == "cyclic":
                self.schedule_obj = torch.optim.lr_scheduler.CyclicLR(optim_obj, **self.optim_config["scheduler_kwargs"])
            else:
                self.schedule_obj = optim_obj
        except AttributeError:
            log.info(f"Scheduler {self.optim_config['scheduler']} not provided or not understood")
            self.schedule_obj = optim_obj
