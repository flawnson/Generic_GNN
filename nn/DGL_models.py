import torch
import torch.nn.functional as F

from abc import ABC
from nn.DGL_layers import GNNLayer
from dgl.nn.pytorch.conv import GraphConv, GATConv, GINConv, SAGEConv, ChebConv, EdgeConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SortPooling, GlobalAttentionPooling, Set2Set


class GenericGNNModel(torch.nn.Module, ABC):
    def __init__(self, config: dict, layer_dict: list, pooling: torch.nn.functional, device: torch.device):
        """
        :param config: Model config file (Python dictionary from JSON)
        :param linear_model: Either None or Linear model stored as torch Module object (only implemented for GCN model)
        :param layer_dict: Dicionary containing layer information including sizes, cacheing, etc.
        :param pooling: Either None or torch pooling objects used between layers in forward propagation
        :param device: torch device object defined in main.py
        """
        super(GenericGNNModel, self).__init__()
        self.config = config
        self.layers = torch.nn.ModuleList([self.factory(info) for info in layer_dict])
        self.pool = pooling if pooling else [None] * len(self.layers)
        self.device = device

    @staticmethod
    def factory(sizes: dict):
        name = sizes.get("name")
        sizes_copy = sizes.copy()
        sizes_copy.pop("name", None)
        return eval(name)(**sizes_copy)

    def forward(self, graph_obj, x):
        z = x
        for layer, pooling in zip(self.layers, self.pool):
            x = layer(graph_obj, x)
            z = x
            x = pooling(graph_obj, x) if pooling else x
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = z
        return x


class GNNModel(GenericGNNModel, ABC):
    # Provide pooling arguments as kwargs (only needed for GlobalAttentionPooling and Set2Set (forward parameters should
    # be provided in the forward function of the model)
    def __init__(self, config: dict, data: torch.tensor, device: torch.device, pooling: str = None, **kwargs):
        super(GNNModel, self).__init__(
            config=config,
            layer_dict=[dict(name=GNNLayer.__name__,
                        in_channels=in_size,
                        out_channels=out_size)
                        for in_size, out_size in zip(config["layer_sizes"], config["layer_sizes"][1:])],
            pooling=[eval(pooling)(kwargs).to(device) for size in config["layer_sizes"][1:]] if pooling else None,
            device=device)
        self.data = data
