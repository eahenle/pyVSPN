import torch
import torch_geometric
import numpy


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
        for _ in range(1, self.mpnn_steps+1):
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
        self.pooling_layer = torch.nn.Linear(hidden_encoding_length, graph_encoding_length, bias=False)
        # Readout layer returns prediction from graph encoding vector
        self.readout_layer = torch.nn.Linear(graph_encoding_length, 1)
    
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


def main():
    # model hyperparameters
    node_encoding_length = 5
    hidden_encoding_length = 7
    graph_encoding_length = 3
    mpnn_steps = 2

    # training parameters
    nb_epochs = 1000 # maximum number of training epochs
    stopping_threshold = 0.001 # loss threshold for breaking out of the training loop

    # instantiate the model
    model = Model(node_encoding_length, hidden_encoding_length, graph_encoding_length, mpnn_steps)

    # make some data
    x = torch.tensor(numpy.random.rand(8,node_encoding_length), dtype=torch.float)
    edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0]
        ],
        dtype=torch.long)
    y = torch.tensor([0.5], dtype=torch.float)
    datum = torch_geometric.data.Data(x = x, edge_index = edge_index, y = y)

    # instantiate objects for auto-differentiation and optimization
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.L1Loss()

    # see what the randomly initialized model predicts (should be very bad)
    print("Before training:")
    y_hat = model(datum)
    print("Prediction:\n{}\n".format(y_hat.item()))

    # run the training loop
    for i in range(1, nb_epochs):
        optimizer.zero_grad() # reset the gradients
        y_hat = model(datum) # make prediction
        loss = loss_func(y_hat, datum.y) # calculate loss
        if loss.item() < stopping_threshold: # evaluate early stopping
            print("Breaking training loop at iteration {}\n".format(i))
            break
        loss.backward() # do back-propagation
        optimizer.step() # update optimizer

    # see what the trained model predicts (should be very good)
    print("After training:")
    y_hat = model(datum)
    print("Prediction:\n{}\n".format(y_hat.item()))


if __name__ == "__main__":
    main()
