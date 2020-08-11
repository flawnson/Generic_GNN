"""This file contains the code used for early stopping regularization for each run type"""

import numpy as np

from typing import List, Dict


class Stop(object):
    def __init__(self, config: Dict):
        self.config = config
        self.states = dict(zip(self.config["early_stop"].keys(), np.zeros(len(self.config["early_stop"]))))
        self.accumulators = dict(zip(self.config["early_stop"].keys(), np.zeros(len(self.config["early_stop"]))))

    def early_stopping(self, value, name):
        if value > self.states[name]:
            self.accumulators[name] += 1
            self.states[name] = value

        assert self.states[name] < self.config.get("early_stop")[name], f'{name} failed to improve for ' \
                                                                     f'{self.config.get("early_stop")[name]} iter,' \
                                                                     f'early stopping current iter'
