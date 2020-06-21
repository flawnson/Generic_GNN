import torch
import torch.nn.functional as F

from abc import ABC
from nn.DGL_layers import GNNLayer


class GNNModel(torch.nn.Module, ABC):
    def __init__(self, names, sizes: list):
        super(GNNModel, self).__init__()
        self.names = names
        self.sizes = sizes
        self.layers = [self.layer_factory(name) for name in names]

    def layer_factory(self, name):
        layer = eval(self.name)
        return layer(self.sizes, self.sizes[1:])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)

