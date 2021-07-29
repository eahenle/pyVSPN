import torch
import torch_geometric

class MPNN(torch_geometric.nn.MessagePassing):
    """
    mpnn = MPNN(node_encoding_length, hidden_encoding_length, mpnn_steps, msg_aggr="add")

    Defines a message-passing neural network that takes graphs having input node encodings of length `node_encoding_length`,
    and hidden node encodings of length `hidden_encoding_length`, which performs `mpnn_steps` of message propagation,
    aggregating messages according to `msg_aggr`.
    """
    
    # constructor
    def __init__(self, node_encoding_length, hidden_encoding_length, mpnn_steps, msg_aggr="add"):
        assert mpnn_steps > 0
        super(MPNN, self).__init__(aggr=msg_aggr)
        self.mpnn_steps = mpnn_steps
        self.pad = torch.nn.ZeroPad2d((0, hidden_encoding_length - node_encoding_length, 0, 0))
    
    # forward-pass behavior
    def forward(self, x, edge_index):
        # pad nodes' feature vectors w/ trailing 0's
        x = self.pad(x)
        # perform mpnn_steps of message propagation (messaging, aggregation, updating)
        for _ in range(self.mpnn_steps):
            x = self.propagate(edge_index, x=x)
        return x, edge_index
    
    # message function
    def message(self, x_j):
        return x_j


class Model(torch.nn.Module):
    """
    model = Model(node_encoding_length, hidden_encoding_length, graph_encoding_length, mpnn_steps)

    Creates a model for graph-level property prediction using an MPNN to develop node encodings and
    a simple pooling/readout affine neural net.
    """

    # constructor
    def __init__(self, node_encoding_length, hidden_encoding_length, graph_encoding_length, mpnn_steps):
        super(Model, self).__init__()
        assert hidden_encoding_length >= node_encoding_length
        # MPNN pads encoding out to hidden encoding length and develops latent representation
        self.mpnn_layers = MPNN(node_encoding_length, hidden_encoding_length, mpnn_steps)
        # Pooling layer reduces node encoding matrix to fixed-length vector
        self.pooling_layer = torch.nn.Linear(hidden_encoding_length, graph_encoding_length, bias=False) ## TODO add ReLU in here
        # Readout layer returns prediction from graph encoding vector
        self.readout_layer = torch.nn.Linear(graph_encoding_length, 1)
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.parameters()) ## TODO learning rate
    
    # forward-pass behavior
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # do message passing
        x, _ = self.mpnn_layers(x, edge_index)
        # do pooling
        x = self.pooling_layer(x)
        x = torch.mean(x, 0)
        # make prediction
        x = self.readout_layer(x)
        return x

    # training routine
    def train(self, data, loss_func, nb_epochs, stopping_threshold):
        for i in range(nb_epochs): # train for up to `nb_epochs` cycles
            loss = 0
            self.optimizer.zero_grad() # reset the gradients (...why?)
            for datum in data:
                y_hat = self(datum) # make prediction
                loss += loss_func(y_hat, datum.y) # accumulate loss ## TODO regularize
            if loss.item() / len(data) < stopping_threshold: # evaluate early stopping
                print("Breaking training loop at iteration {}\n".format(i))
                break
            if i % 250 == 0:
                print(f"Epoch\t{i}\t|\tLoss\t{loss/len(data)}")
            loss.backward() # do back-propagation to get gradients
            self.optimizer.step() # update weights