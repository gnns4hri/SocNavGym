# Implementation of the message passing graph neural network according to the article in:
# https://arxiv.org/pdf/1704.01212.pdf

import numpy as np
import torch as th
import dgl
from dgl.nn import NNConv


class MPNN(th.nn.Module):
    def __init__(self, num_feats, n_classes, hidden, num_edge_feats, final_activation, aggregator_type='mean',
                 bias=True, residual=False, norm=None, activation=None):
        super(MPNN, self).__init__()
        self._num_feats = num_feats
        self._n_classes = n_classes
        self._num_hiden_features = hidden
        self._num_edge_feats = num_edge_feats
        self._aggregator = aggregator_type
        self._activation = activation
        self._final_activation = final_activation
        self._norm = norm

        # Input layer
        edge_function = self.edge_function(self._num_edge_feats, self._num_feats*self._num_hiden_features[0])
        self.NNconv_input = NNConv(self._num_feats, self._num_hiden_features[0], edge_function, self._aggregator,
                                   residual, bias)

        # Hidden layers
        self.layers = th.nn.ModuleList()
        for idx in range(1, len(self._num_hiden_features)):
            edge_function = self.edge_function(self._num_edge_feats,
                                          self._num_hiden_features[idx-1]*self._num_hiden_features[idx])
            self.layers.append(NNConv(self._num_hiden_features[idx-1], self._num_hiden_features[idx], edge_function,
                                      self._aggregator, residual, bias))

        # Output layer
        edge_function = self.edge_function(self._num_edge_feats, self._num_hiden_features[-1]*self._n_classes)
        self.NNConv_output = NNConv(self._num_hiden_features[-1], self._n_classes, edge_function, self._aggregator,
                                    residual, bias)

    @staticmethod
    def edge_function(f_in, f_out):
        a = int(f_in*0.666 + f_out*0.334)
        b = int(f_in*0.334 + f_out*0.666)
        return th.nn.Sequential(
            th.nn.Linear(f_in, a),
            th.nn.ReLU(),
            th.nn.Linear(a, b),
            th.nn.ReLU(),
            th.nn.Linear(b, f_out)
        )

    def forward(self, graph, feat, efeat):
        x = self.NNconv_input(graph, feat, efeat)

        # activation
        if self._activation is not None:
            x = self._activation(x)
        # normalization
        if self._norm is not None:
            x = self._norm(x)

        for idx, layer in enumerate(self.layers, 1):
            x = layer(graph, x, efeat)
            # activation
            if self._activation is not None:
                x = self._activation(x)
            # normalization
            if self._norm is not None:
                x = self._norm(x)

        x = self.NNConv_output(graph, x, efeat)
        if self._final_activation is not None:
            logits = self._final_activation(x)
        else:
            logits = x

        return logits
