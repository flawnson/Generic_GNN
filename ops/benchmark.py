"""This file contains the code for connecting the tuning and training pipelines to form the benchmarking pipeline"""

import torch
from ops.tune import Tuner
from ops.train import Trainer


class Benchmarker():
    def __init__(self, config, device):
        self.benchmarking_config = config["benchmarking_config"]
        self.device = device

    def run_tune(self):
        Tuner(json_data["tune_config"], dataset, device).run_tune()

    def run_train(self):
        Trainer(json_data["train_config"], dataset, model, device).run_train()
