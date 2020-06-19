import dgl
import torch

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    # XXX: Reference for self; u: source node, v: destination node, e edges among those nodes
    def __init__(self, in_size, out_size, activation, weight=True, bias=True, **kwargs):
        super(GNNLayer, self).__init__()
        self._activation = activation
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.bias: bool = bias

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_size, out_size), requires_grad=True)
        else:
            self.register_parameter('weight', None)

    def func_factory(self):
        # func = fn.
        return None

    def reset_parameters(self):
        # Obligatory parameter reset method
        pass

    def forward(self, graph_obj, feature, weight=True):
        # local_scope needed to ensure that the stored data (in messages) doesn't accumulate
        with graph_obj.local_scope():
            if self._norm == 'both':
                degs = graph_obj.out_degrees().to(feature.device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feature.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat = feature * norm

            if weight is not None:
                if self.weight is not None:
                    raise AssertionError('External weight is provided while at the same time the'
                                         ' module has defined its own weight parameter. Please'
                                         ' create the module with flag weight=False.')
            else:
                weight = self.weight

            graph_obj.ndata["feature_name"] = feature
            graph_obj.update_all()
            h = graph_obj.ndata["h"]
            if self.activation is not None:
                h = self._activation(h)

        return self.linear(h)
