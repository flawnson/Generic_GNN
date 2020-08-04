import numpy as np

from


class Stop():
    def __init__(self, metric, names: list[str]):
        self.metric = metric
        self.names = names

    @property
    def init(self):
        return dict(zip(self.names, np.zeros(len(self.names))))

    @init.setter
    def early_stopper(self, value):
        if value > loss_state:
            loss_no_improve += 1

        if loss_no_improve > config.get("loss_patience"):
            raise AssertionError(f'Loss failed to decrease for {config["loss_patience"]} iter, early stopping current iter')
