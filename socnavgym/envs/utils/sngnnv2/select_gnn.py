import torch
import torch.nn as nn
from nets.rgcnDGL import RGCN
from nets.gat import GAT
from nets.mpnn_dgl import MPNN
import dgl

class SELECT_GNN(nn.Module):
    def __init__(self, num_features, num_edge_feats, n_classes, num_hidden, gnn_layers, dropout,
                 activation, final_activation, gnn_type, num_heads, num_rels, num_bases, g, residual,
                 aggregator_type, attn_drop, concat=True, bias=True, norm=None, alpha=0.12):
        super(SELECT_GNN, self).__init__()

        self.activation = activation
        self.gnn_type = gnn_type
        if final_activation == 'relu':
            self.final_activation = torch.nn.ReLU()
        elif final_activation == 'tanh':
            self.final_activation = torch.nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None

        self.attn_drop = attn_drop
        self.num_rels = num_rels
        self.residual = residual
        self.aggregator = aggregator_type
        self.num_bases = num_bases
        self.n_classes = n_classes
        self.num_hidden = num_hidden
        self.gnn_layers = gnn_layers
        self.num_features = num_features
        self.num_edge_feats = num_edge_feats
        self.dropout = dropout
        self.bias = bias
        self.norm = norm
        self.g = g
        self.num_heads = num_heads
        self.concat = concat
        self.alpha = alpha

        if self.gnn_type == 'rgcn':
            # print("GNN being used is RGCN")
            self.gnn_object = self.rgcn()
        elif self.gnn_type == 'gat':
            # print("GNN being used is GAT")
            self.gnn_object = self.gat()
        elif self.gnn_type == 'mpnn':
            # print("GNN being used is MPNN")
            self.gnn_object = self.mpnn()

    def rgcn(self):
        return RGCN(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_rels,
                    self.activation, self.final_activation, self.dropout, self.num_bases)

    def gat(self):
        return GAT(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_heads,
                   self.activation, self.final_activation,  self.dropout, self.attn_drop, self.alpha, self.residual)

    def mpnn(self):
        return MPNN(self.num_features, self.n_classes, self.num_hidden, self.num_edge_feats, self.final_activation,
                    self.aggregator, self.bias, self.residual, self.norm, self.activation)

    def forward(self, data, g, efeat):
        if self.gnn_type == 'mpnn':
            x = self.gnn_object(g, data, efeat)
        else:
            x = self.gnn_object(data, g)
        logits = x
        base_index = 0
        batch_number = 0
        unbatched = dgl.unbatch(self.g)
        output = torch.Tensor(size=(len(unbatched), 2))
        for g in unbatched:
            num_nodes = g.number_of_nodes()
            output[batch_number, :] = logits[base_index, :]  # Output is just the room's node
            base_index += num_nodes
            batch_number += 1
        return output
