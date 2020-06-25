import numpy as np
import torch


class Holdout:
    def __init__(self, data_config, dataset):
        super(Holdout, self)
        self.data_config = data_config
        self.dataset = dataset

    def split(self):
        self.data_config["split_sizes"]
        self.data_config["num_splits"]

