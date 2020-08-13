"""This file contains the code used for early stopping regularization for each run type"""

import numpy as np

from typing import List, Dict


class Stop(object):
    def __init__(self, config: Dict):
        self.early_stop = config["early_stop"]
        self.states = dict(zip(self.early_stop.keys(), np.zeros(len(self.early_stop))))
        self.accumulators = dict(zip(self.early_stop.keys(), np.zeros(len(self.early_stop))))

    def early_stopping(self, value: float, name: str, greater: bool = True):
        if greater:
            if value > self.config["early_stop"]["name"]:
                self.accumulators[name] += 1
                self.states[name] = value
        else:
            if value < self.config["early_stop"]["name"]:
                self.accumulators[name] += 1
                self.states[name] = value

        assert self.accumulators[name] < self.early_stop[name], f'{name} failed to improve for {self.early_stop[name]}' \
                                                                f'iter, early stopping current iter'
