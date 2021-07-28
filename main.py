""" TODO
- arg parse
- load data
- train
- test
"""

import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
import numpy as np


class MPNN(MessagePassing):
    def __init__(self, node_encoding_length, hidden_encoding_length, mpnn_steps, msg_aggr="add"):
        super(MPNN, self).__init__(aggr=msg_aggr)
        self.mpnn_steps = mpnn_steps
        self.hidden_encoding_length = hidden_encoding_length
        self.node_encoding_length = node_encoding_length
    
    def forward(self, x, edge_index):
        pad = torch.nn.ZeroPad2d((0, self.hidden_encoding_length - self.node_encoding_length, 0, 0))
        x = pad(x)
        x = self.propagate(edge_index, x=x)
        for _ in range(2, self.mpnn_steps+1):
            x = self.propagate(edge_index, x=x)
        return x, edge_index
    
    def message(self, x_j):
        return x_j


class VSPN(torch.nn.Module):
    def __init__(self, node_encoding_length, hidden_encoding_length, graph_encoding_length, mpnn_steps):
        super(VSPN, self).__init__()
        assert hidden_encoding_length >= node_encoding_length
        # MPNN pads encoding out to hidden encoding length and develops latent representation
        self.mpnn_layers = MPNN(node_encoding_length, hidden_encoding_length, mpnn_steps)
        # Pooling layer reduces node encoding matrix to fixed-length vector
        self.pooling_layer = torch.nn.Linear(hidden_encoding_length, graph_encoding_length, bias=False)
        # Readout layer returns prediction from graph encoding vector
        self.readout_layer = torch.nn.Linear(graph_encoding_length, 1)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x, _ = self.mpnn_layers(x, edge_index)
        x = self.pooling_layer(x)
        x = torch.mean(x, 0)
        x = self.readout_layer(x)
        return x


def main():
    node_encoding_length = 5
    hidden_encoding_length = 7
    graph_encoding_length = 3
    mpnn_steps = 2
    vspn = VSPN(node_encoding_length, hidden_encoding_length, graph_encoding_length, mpnn_steps)

    x = torch.tensor(np.random.rand(8,node_encoding_length), dtype=torch.float)
    edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0]
        ],
        dtype=torch.long)
    y = torch.tensor([0.5], dtype=torch.float)
    datum = torch_geometric.data.Data(x = x, edge_index = edge_index, y = y)

    parameters = vspn.parameters()
    optimizer = torch.optim.Adam(parameters)
    crit = torch.nn.L1Loss()

    print("Before training:")
    y_hat = vspn(datum)
    print("Prediction:\n{}\n".format(y_hat.item()))

    for i in range(1, 1000):
        optimizer.zero_grad()
        y_hat = vspn(datum)
        loss = crit(y_hat, datum.y)
        if loss.item() < 0.001:
            print("Breaking training loop at iteration {}\n".format(i))
            break
        loss.backward()
        optimizer.step()

    print("After training:")
    y_hat = vspn(datum)
    print("Prediction:\n{}\n".format(y_hat.item()))

if __name__ == "__main__":
    main()
