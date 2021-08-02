import torch
import torch_geometric

class MPNN(torch_geometric.nn.MessagePassing):
    """
    mpnn = MPNN(mpnn_steps, msg_aggr)

    Defines a message-passing neural network which performs `mpnn_steps` of message propagation,
    aggregating messages according to `msg_aggr`.
    """
    
    # constructor
    def __init__(self, mpnn_steps, msg_aggr):
        assert mpnn_steps > 0
        super(MPNN, self).__init__(aggr=msg_aggr)
        self.mpnn_steps = mpnn_steps
    
    # forward-pass behavior
    def forward(self, x, edge_index):
        # perform mpnn_steps of message propagation (messaging, aggregation, updating)
        for _ in range(self.mpnn_steps):
            x = self.propagate(edge_index, x=x)
        return x, edge_index
    
    # message function
    def message(self, x_j):
        return x_j

    # update function
    def update(self, messages, x):
        x = (x + messages) / 2
        return x


class Model(torch.nn.Module):
    """
    model = Model(node_encoding_length, hidden_encoding_length, mpnn_steps)

    Creates a model for graph-level property prediction using an MPNN to develop node encodings and
    a simple readout/prediction affine neural net.
    """

    # constructor
    def __init__(self, node_encoding_length, hidden_encoding_length, mpnn_steps, s2s_steps, mpnn_aggr="mean"):
        super(Model, self).__init__()
        assert hidden_encoding_length >= node_encoding_length
        # ReLU for nonlinear activation steps
        self.relu = torch.nn.ReLU()
        # input layer transforms encoding to hidden encoding length 
        self.input_layer = torch.nn.Linear(node_encoding_length, hidden_encoding_length)
        # MPNN develops latent representation
        self.mpnn_layers = MPNN(mpnn_steps, mpnn_aggr)
        # readout layer reduces node encoding matrix to fixed-length vector
        self.readout_layer = torch_geometric.nn.Set2Set(hidden_encoding_length, s2s_steps)
        # prediction layer returns prediction from graph encoding vector
        self.prediction_layer = torch.nn.Linear(2 * hidden_encoding_length, 1)
    
    # forward-pass behavior
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # transform input
        x = self.input_layer(x)
        x = self.relu(x)
        # do message passing
        x, _ = self.mpnn_layers(x, edge_index)
        # do readout
        x = self.readout_layer(x, data.batch)
        # make prediction
        x = self.prediction_layer(x)
        return x
