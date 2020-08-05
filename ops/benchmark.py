"""This file contains the code for connecting the tuning and training pipelines to form the benchmarking pipeline"""

import torch


class Benchmarker():
    def __init__(self, benchmarking_config, device):
        self.benchmarking_config = benchmarking_config
        self.device = device