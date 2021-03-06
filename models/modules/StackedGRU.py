import torch.nn as nn
import torch

class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])  # (B, 2*H)
            input = h_1_i  # (B, 2*H)
            if i + 1 != self.num_layers:  # 非最后一层
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input, h_1
