"""This file contains the code for connecting the tuning and training pipelines to form the benchmarking pipeline"""

import torch

from ops.tune import Tuner
from ops.train import Trainer
from read.preprocessing import GenericDataset


class Benchmarker:
    def __init__(self, config: dict, dataset: GenericDataset, device: torch.device):
        self.config = config
        self.dataset = dataset
        self.device = device

    def run_tune(self) -> dict:
        return Tuner(self.config, self.dataset, self.device).run_tune()

    def run_train(self, best_config: dict):
        return Trainer(best_config, self.dataset, self.device).run_train()

    def run_benchmark(self):
        return self.run_train(self.run_tune())
