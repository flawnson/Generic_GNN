import torch


class Benchmarker():
    def __init__(self, benchmarking_config, device):
        self.benchmarking_config = benchmarking_config
        self.device = device