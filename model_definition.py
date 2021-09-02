import torch
import torch_geometric

class MPNN(torch_geometric.nn.MessagePassing):
    """
    mpnn = MPNN(mpnn_steps, msg_aggr)

    Defines a message-passing neural network which performs `mpnn_steps` of message propagation,
    aggregating messages according to `msg_aggr`.
    """
    
    # constructor
    def __init__(self, mpnn_steps, input_size, hidden_size, msg_aggr, update_func):
        assert mpnn_steps > 0
        #assert hidden_size > input_size ## TODO ?
        assert update_func == "mean" ## TODO add GRU update option
        super(MPNN, self).__init__(aggr=msg_aggr)
        self.mpnn_steps = mpnn_steps
        self.update_func = update_func
        self.pad = torch.nn.ZeroPad2d((0, hidden_size - input_size, 0, 0))
    
    # forward-pass behavior
    def forward(self, x, edge_index):
        # pad x -> h
        h = self.pad(x)
        # perform mpnn_steps of message propagation (messaging, aggregation, updating)
        for _ in range(self.mpnn_steps):
            h = self.propagate(edge_index, h=h)
        return h, edge_index
    
    # message function
    def message(self, h_j):
        return h_j

    # update function: mean
    def update_mean(self, m, h):
        return (h + m) / 2

    # update function switch
    def update(self, m, h):
        if self.update_func == "mean":
            return self.update_mean(m, h)
        elif self.update_func == "mgu":
            return self.update_mgu(m, h)
        


class Model(torch.nn.Module):
    """
    model = Model(node_encoding_length, args)

    Creates a model for graph-level property prediction using an MPNN to develop node encodings,
    Set2Set for graph-level readout, and an affine prediction layer.
    """

    # constructor
    def __init__(self, node_feature_length, args):
        # unpack args
        input_encoding_length = args.input_encoding
        hidden_encoding_length = args.hidden_encoding
        mpnn_steps = args.mpnn_steps
        mpnn_aggr = args.mpnn_aggr
        mpnn_update = args.mpnn_update
        # initialize base class
        super(Model, self).__init__()
        # ReLU for nonlinear activation
        self.relu = torch.nn.ReLU()
        # input layer transforms encoding to hidden encoding length 
        self.input_layer = torch.nn.Linear(node_feature_length, input_encoding_length)
        # MPNN develops latent representation
        self.mpnn_layers = MPNN(mpnn_steps, input_encoding_length, hidden_encoding_length, mpnn_aggr, mpnn_update)
        # prediction layer returns prediction from graph encoding vector
        self.prediction_layer = torch.nn.Linear(hidden_encoding_length, 1)
    
    # forward-pass behavior
    def forward(self, datum):
        # transform input
        x = self.input_layer(datum.x)
        # ReLU activation
        x = self.relu(x)
        # do message passing
        x, _ = self.mpnn_layers(x, datum.edge_index)
        # do readout to graph-level encoding vector
        x = torch.mean(x, 0)
        # make prediction
        return self.prediction_layer(x)
