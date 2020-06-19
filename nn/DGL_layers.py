import dgl
import torch

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    # XXX: Reference for self; u: source node, v: destination node, e edges among those nodes
    def __init__(self, in_size, out_size, bias, activation, **kwargs):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.bias: bool = bias
        self.activation = activation

    def func_factory(self):
        func = fn.
        return func

    def forward(self, graph_obj, feature):
        # local_scope needed to ensure that the stored data (in messages) doesn't accumulate
        with graph_obj.local_scope():
            graph_obj.ndata["feature_name"]
            graph_obj.update_all()
            graph_obj.send_and_recv()

            h = graph_obj.ndata["h"]

            if self.activation is not None:


            return self.linear(h)
