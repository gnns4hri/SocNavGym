import torch
import torch.nn as nn
from torch.nn import functional as F


# Define the RNN model (using LSTM here)
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type, linear_layers=[], activation = 'linear', context_vars = 0, dropout = 0.0):
        super(RNNModel, self).__init__()
        if rnn_type == "GRU":
            self.layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "LSTM":
            self.layer = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout = dropout)
        self.fc_layers = []
        linear_size = hidden_size+context_vars
        for l in linear_layers:
            self.fc_layers.append(nn.Linear(linear_size, l))
            linear_size = l
            self.fc_layers.append(nn.LeakyReLU())
        self.fc_layers.append(nn.Linear(linear_size,1))
        self.fc = self.fc_layers[-1]
        if len(linear_layers)>0:
            self.fc_layers = nn.Sequential(*self.fc_layers)

        # self.fc = nn.Linear(hidden_size+context_vars, 128)
        # self.fc2 = nn.Linear(128, 1)
        if activation == 'sigmoid':
            self.correct_output = False
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.correct_output = True
            self.activation = nn.Tanh()
        else:
            self.correct_output = False
            self.activation = None
        self.context_vars = context_vars

    def forward(self, x, slengths):
        # x: (batch, sequence_length, input_size)
        out, _ = self.layer(x)

        out = out[torch.arange(out.shape[0]), slengths]
        if self.context_vars>0:
            out = torch.concat((out, x[:,0, -self.context_vars:]), axis=1)

        for layer in self.fc_layers:
            out = layer(out)

        if self.activation is not None:
            out = self.activation(out)
        if self.correct_output:
            out = (out+1.)/2.
        
        return out
