import dgl
import torch

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    # XXX: Reference for self; u: source node, v: destination node, e edges among those nodes
    # Generic GNN layer can be modified with DGL's built in tools (currently implemented as GCN)
    def __init__(self, in_size, out_size, weight=True, bias=True):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.bias: bool = bias
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_size, out_size), requires_grad=True)
        else:
            self.register_parameter('weight', None)

    def reset_parameters(self):
        # Obligatory parameter reset method
        pass

    def forward(self, graph_obj, feature, weight=True):
        # local_scope needed to ensure that the stored data (in messages) doesn't accumulate
        # When implementing a layer you have a choice of initializing your own weights and matmul, or using nn.Linear
        # For performance reasons, DGL's implementation performs operations in order according to input/output size
        # It should be possible however, to use matmul(features, weights) or nn.Linear(features) anyplace anytime
        with graph_obj.local_scope():
            if self._norm == 'both':
                degs = graph_obj.out_degrees().to(feature.device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feature.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat = feature * norm

            # feat = torch.matmul(feat, weight)
            graph_obj.srcdata['h'] = feat
            graph_obj.update_all(fn.copy_src(src='h', out='m'),
                                 fn.sum(msg='m', out='h'))
            feat = graph_obj.dstdata['h']

            if self.bias is not None:
                feat += self.bias

        return self.linear(feat)
