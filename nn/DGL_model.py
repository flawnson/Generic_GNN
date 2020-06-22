import torch
import torch.nn.functional as F

from abc import ABC
from nn.DGL_layers import GNNLayer
from dgl.nn.pytorch.conv import GraphConv, GATConv, GINConv, SAGEConv, ChebConv, EdgeConv


class GenericGNNModel(torch.nn.Module, ABC):
    def __init__(self, config: dict, linear_model: torch.nn.Module, sizes, pooling, device):
        """
        :param config: Model config file (Python dictionary from JSON)
        :param linear_model: Either None or Linear model stored as torch Module object (only implemented for GCN model)
        :param sizes: Dicionary containing layer information including sizes, cacheing, etc.
        :param pooling: Either None or torch pooling objects used between layers in forward propagation
        :param device: torch device object defined in main.py
        """
        super(GenericGNNModel, self).__init__()
        self.config = config
        self.linear_model = linear_model
        self.layers = torch.nn.ModuleList([self.factory(size) for size in sizes])
        self.pool = pooling if pooling else [None] * len(self.layers)
        self.device = device

    @staticmethod
    def factory(sizes: dict):
        name = sizes.get("name")
        sizes_copy = sizes.copy()
        sizes_copy.pop("name", None)
        return eval(name)(**sizes_copy)

    def forward(self, x):
        z = x
        for layer in self.layers:
            x = layer(x)
            z = x
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = z
        return x


class GNNModel(GenericGNNModel):
    def __init__(self, config, data: torch.tensor, sizes: list, device, pooling: str = None, **kwargs):
        super(GNNModel, self).__init__(
            config=config,
            layer_dict=[dict(name=GNNLayer.__name__,
                        in_channels=in_size,
                        out_channels=out_size,
                        improved=kwargs["improved"],
                        cached=kwargs["cached"])
                   for in_size, out_size in zip(sizes, sizes[1:])],
            pooling=[eval(pooling)(in_channels=size).to(device) for size in sizes[1:]] if pooling else None,
            device=device)
        self.data = data
