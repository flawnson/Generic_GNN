""" This file contains the code used for early stopping regularization for each run type works for both greater than
    and less than situations """

import numpy as np


class Stop(object):
    def __init__(self, config: dict):
        self.early_stop = config["early_stop"]
        self.states = dict(zip(self.early_stop.keys(), np.zeros(len(self.early_stop))))
        self.accumulators = dict(zip(self.early_stop.keys(), np.zeros(len(self.early_stop))))

    def early_stopping(self, value: float, name: str, greater: bool = True):
        # The greater argument corresponds with the direction with which improvement is determined, hence accuracy
        # should be greater (True) and loss should not be greater (False)
        more = value if greater else self.states[name]
        less = self.states[name] if greater else value

        if more > less:
            self.accumulators[name] += 1
            self.states[name] = value

        assert self.accumulators[name] < self.early_stop[name], f'{name} failed to improve for {self.early_stop[name]}' \
                                                                f'iter, early stopping current iter'
