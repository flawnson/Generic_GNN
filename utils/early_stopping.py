import numpy as np

from typing import List, Dict


class Stop(object):
    def __init__(self, config: Dict):
        self.config = config
        self.states = dict(zip(self.config["early_stop"].keys(), np.zeros(len(self.config["early_stop"].keys()))))
        self.accumulators = dict(zip(self.config["early_stop"].keys(), np.zeros(len(self.config["early_stop"].keys()))))

    def early_stopping(self, value, name):
        if value > self.states[name]:
            self.accumulators[name] += 1
            self.states[name] = value

        assert self.states[name] > self.config.get("loss_patience"), f'{name} failed to improve for ' \
                                                                     f'{self.config["patience"]} iter,' \
                                                                     f'early stopping current iter'
