import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu
from torch_geometric.nn import GATConv, Linear
from .transformer_block import TransformerBlock

class GNNModel(nn.Module):
    def __init__(self, gnn_input_channels, gnn_hidden_channels, gnn_heads, gnn_output, gnn_concat):
        super(GNNModel,self).__init__()
        self.layers_gat = nn.ModuleList()
        self.layers_lin = nn.ModuleList()

        self.layers_gat.append(GATConv(gnn_input_channels, gnn_hidden_channels[0], gnn_heads[0], gnn_concat, add_self_loops=True))
        self.layers_lin.append(Linear(gnn_input_channels, gnn_hidden_channels[0]))

        heads = gnn_heads[0]
        for idx in range(len(gnn_hidden_channels) - 1):
            input_dim = gnn_hidden_channels[idx]
            if gnn_concat:
                input_dim*=heads
            output_dim = gnn_hidden_channels[idx + 1]
            heads = gnn_heads[idx + 1]            

            self.layers_gat.append(GATConv(input_dim, output_dim, heads, gnn_concat, add_self_loops=True))
            self.layers_lin.append(Linear(input_dim, output_dim))

        if gnn_concat:
            output_dim*=heads

        self.layers_gat.append(GATConv(output_dim, gnn_output, heads=1, concat=False, add_self_loops=True))
        self.layers_lin.append(Linear(output_dim, gnn_output))


    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers_gat):
            x = layer(x, edge_index)#+self.layers_lin[i](x)
            if i < len(self.layers_gat) - 1:
                x = leaky_relu(x, negative_slope=0.1)
        return x

class RNNModel(nn.Module):
    def __init__(self, rnn_type, rnn_input, rnn_hidden_channels, num_layers, batch_first, dropout):
        super(RNNModel,self).__init__()

        if rnn_type == 'GRU':
            self.rnn_block = nn.GRU(rnn_input, rnn_hidden_channels, num_layers, batch_first=batch_first, dropout=dropout)
        elif rnn_type=='LSTM':
            self.rnn_block = nn.LSTM(rnn_input, rnn_hidden_channels, num_layers, batch_first=batch_first, dropout=dropout)
        else:
            print(f'{rnn_type} is not a valid RNN type')
            exit()

    def forward(self, x, x_len):
        rnn_output, _ = self.rnn_block(x)
        out = rnn_output[torch.arange(rnn_output.shape[0]), x_len - 1]
        return out

class HybridModel(nn.Module):
    def __init__(self, num_layers, gnn_input, gnn_output, rnn_hidden_channels, gnn_hidden_channels, rnn_type, gnn_heads, gnn_concat, 
                 linear_layers=[], rnn_activation = 'linear', context_vars = 0, metrics_vars = 0, rnn_dropout = 0.0, only_gnn = False, only_metrics = False):
        super(HybridModel,self).__init__()

        self.gnn_output = gnn_output
        self.num_layers = num_layers
        self.only_gnn = only_gnn
        self.only_metrics = only_metrics
        self.context_vars = context_vars
        self.metrics_vars = metrics_vars
        self.scenario_vars = context_vars+metrics_vars

        self.gnn_block = GNNModel(gnn_input, gnn_hidden_channels, gnn_heads, gnn_output, gnn_concat)
        
        self.contextNorm = nn.LayerNorm(self.context_vars)

        if not self.only_gnn:
            self.metricsNorm = nn.LayerNorm(self.metrics_vars)
        if not self.only_metrics:
            self.gnnNorm = nn.LayerNorm(gnn_output)

        self.defineSeqBlock(rnn_type, rnn_hidden_channels, linear_layers, rnn_activation, rnn_dropout)

    def defineSeqBlock(self, rnn_type, rnn_hidden_channels, linear_layers, rnn_activation, rnn_dropout):

        if self.only_metrics:
            input_vars = self.scenario_vars
        elif self.only_gnn:
            input_vars = self.gnn_output+self.context_vars
        else:
            input_vars = self.gnn_output+self.scenario_vars

        if rnn_type=='GRU' or rnn_type=='LSTM':
            self.seq_block = RNNModel(rnn_type, input_vars, rnn_hidden_channels, self.num_layers, batch_first=True, dropout=rnn_dropout)
        elif rnn_type=='Transformer':
            self.seq_block = TransformerBlock(input_vars, d_model=rnn_hidden_channels)

        self.fc_layers = nn.ModuleList()
        linear_size = rnn_hidden_channels+self.context_vars
        for l in linear_layers:
            self.fc_layers.append(nn.Linear(linear_size, l))
            linear_size = l
            self.fc_layers.append(nn.LeakyReLU())

        self.fc_layers.append(nn.Linear(linear_size,1))
        if len(linear_layers)>0:
            self.mlp = nn.Sequential(*self.fc_layers)

        if rnn_activation == 'sigmoid':
            self.correct_output = False
            self.activation = nn.Sigmoid()
        elif rnn_activation == 'tanh':
            self.correct_output = True
            self.activation = nn.Tanh()
        else:
            self.correct_output = False
            self.activation = None
        self.context_vars = self.context_vars


    def forward(self, batch_data, metrics, slengths):

        if not self.only_metrics:
            gnn_output = self.gnn_block(batch_data.x, batch_data.edge_index)

            robot_node = batch_data.ptr[:-1] # index of the first node of each graph

            scenarios = gnn_output[robot_node]

            seqs = torch.split(scenarios, slengths.tolist())
            x_seq = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
            x_seq = self.gnnNorm(x_seq)

        # WITH METRICS
        metrics_orig_seq = torch.split(metrics, slengths.tolist())
        metrics_orig_seq = torch.nn.utils.rnn.pad_sequence(metrics_orig_seq, batch_first=True)
        metrics_seq = metrics_orig_seq[:,:,:self.metrics_vars]
        context_seq = metrics_orig_seq[:,:,-self.context_vars:]
        context_seq_norm = self.contextNorm(context_seq)

        if self.only_metrics:
            metrics_seq_norm = self.metricsNorm(metrics_seq)
            x_seq = torch.cat((metrics_seq_norm, context_seq_norm), dim=2)
        elif self.only_gnn:
            x_seq = torch.cat((x_seq, context_seq_norm), dim=2)
        else:
            metrics_seq_norm = self.metricsNorm(metrics_seq)
            x_seq = torch.cat((x_seq, metrics_seq_norm, context_seq_norm), dim=2)


        out = self.seq_block(x_seq, slengths)

        if self.context_vars > 0:
            batch_context = context_seq[:, 0, :]
            out = torch.concat((out, batch_context), axis=1)

        for layer in self.fc_layers:
            out = layer(out)

        if self.activation is not None:
            out = self.activation(out)
            
        if hasattr(self, 'correct_output') and self.correct_output:
            out = (out + 1.) / 2.
        
        return out
