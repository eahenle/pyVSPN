import torch
import torch_geometric

class MPNN(torch_geometric.nn.MessagePassing):
    """
    mpnn = MPNN(mpnn_steps, msg_aggr)

    Defines a message-passing neural network which performs `mpnn_steps` of message propagation,
    aggregating messages according to `msg_aggr`.
    """
    
    # constructor
    def __init__(self, mpnn_steps, hidden_size, msg_aggr, update_func):
        assert mpnn_steps > 0
        super(MPNN, self).__init__(aggr=msg_aggr)
        self.mpnn_steps = mpnn_steps
        assert update_func == "mean" or update_func == "mgu"
        self.update_func = update_func
        if update_func == "mgu":
            self.W_f = torch.nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
            self.U_f = torch.nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
            self.b_f = torch.nn.Parameter(torch.randn(hidden_size), requires_grad=True)
            self.W_h = torch.nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
            self.U_h = torch.nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
            self.b_h = torch.nn.Parameter(torch.randn(hidden_size), requires_grad=True)
            self.sigma_g = torch.nn.Sigmoid()
            self.phi_h = torch.nn.Tanh()
    
    # forward-pass behavior
    def forward(self, x, edge_index):
        # perform mpnn_steps of message propagation (messaging, aggregation, updating)
        h = x
        for _ in range(self.mpnn_steps):
            h = self.propagate(edge_index, h=h)
        return h, edge_index
    
    # message function
    def message(self, h_j):
        return h_j

    # update function: mean
    def update_mean(self, m, h):
        return (h + m) / 2

    # update function: mgu
    def update_mgu(self, m, h):
        Wfm = torch.matmul(self.W_f, torch.transpose(m, 0, 1))
        Ufh = torch.matmul(self.U_f, torch.transpose(h, 0, 1))
        f_t = torch.transpose(Wfm + Ufh, 0, 1)
        for i in range(f_t.shape[0]):
            f_t[i,:] += self.b_f
        f_t = self.sigma_g(f_t)
        Whm = torch.matmul(self.W_h, torch.transpose(m, 0, 1))
        Uhm = torch.matmul(self.U_h, torch.transpose(f_t * h, 0, 1))
        h_hat_t = torch.transpose(Whm + Uhm, 0, 1)
        for i in range(h_hat_t.shape[0]):
            h_hat_t[i,:] += self.b_h
        h_hat_t = self.phi_h(h_hat_t)
        return (1 - f_t) * h + f_t * h_hat_t

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
    def __init__(self, node_encoding_length, args):
        # unpack args
        hidden_encoding_length = args.node_encoding
        mpnn_steps = args.mpnn_steps
        s2s_steps = args.s2s_steps
        # initialize base class
        super(Model, self).__init__()
        # validate hyperparameters
        assert hidden_encoding_length >= node_encoding_length
        assert s2s_steps >= 2
        # ReLU for nonlinear activation
        self.relu = torch.nn.ReLU()
        # input layer transforms encoding to hidden encoding length 
        self.input_layer = torch.nn.Linear(node_encoding_length, hidden_encoding_length)
        # MPNN develops latent representation
        self.mpnn_layers = MPNN(mpnn_steps, hidden_encoding_length, args.mpnn_aggr, args.mpnn_update)
        # readout layer reduces node encoding matrix to fixed-length vector
        self.readout_layer = torch_geometric.nn.Set2Set(hidden_encoding_length, s2s_steps)
        # prediction layer returns prediction from graph encoding vector
        self.prediction_layer = torch.nn.Linear(2 * hidden_encoding_length, 1)
    
    # forward-pass behavior
    def forward(self, x, edge_index, batch):
        # transform input
        x = self.input_layer(x)
        # ReLU activation
        x = self.relu(x)
        # do message passing
        x, _ = self.mpnn_layers(x, edge_index)
        # do readout with Set2Set
        x = self.readout_layer(x, batch)
        # make prediction
        return self.prediction_layer(x)
