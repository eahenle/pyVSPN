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
    def __init__(self, node_encoding_length, hidden_encoding_length, mpnn_steps, msg_aggr):
        assert mpnn_steps > 0
        assert hidden_encoding_length >= node_encoding_length
        super(MPNN, self).__init__(aggr=msg_aggr)
        self.mpnn_steps = mpnn_steps
        self.lin = torch.nn.Linear(node_encoding_length, hidden_encoding_length)
        self.relu = torch.nn.ReLU()
    
    # forward-pass behavior
    def forward(self, x, edge_index):
        # non-linearly transform features into hidden encodings for message passing
        x = self.lin(x)
        x = self.relu(x)
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
    model = Model(node_encoding_length, hidden_encoding_length, graph_encoding_length, mpnn_steps)

    Creates a model for graph-level property prediction using an MPNN to develop node encodings and
    a simple pooling/readout affine neural net.
    """

    # constructor
    def __init__(self, node_encoding_length, hidden_encoding_length, graph_encoding_length, mpnn_steps, mpnn_aggr="mean"):
        super(Model, self).__init__()
        assert hidden_encoding_length >= node_encoding_length
        # MPNN transforms encoding to hidden encoding length and develops latent representation
        self.mpnn_layers = MPNN(node_encoding_length, hidden_encoding_length, mpnn_steps, mpnn_aggr)
        # ReLU for nonlinearity after message-passing
        self.relu = torch.nn.ReLU()
        # Pooling layer reduces node encoding matrix to fixed-length vector
        self.pooling_layer = torch.nn.Linear(hidden_encoding_length, graph_encoding_length, bias=False)
        # Readout layer returns prediction from graph encoding vector
        self.readout_layer = torch.nn.Linear(graph_encoding_length, 1)
    
    # forward-pass behavior
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # do message passing
        x, _ = self.mpnn_layers(x, edge_index)
        # activate through ReLU
        x = self.relu(x)
        # do pooling
        x = self.pooling_layer(x)
        x = torch.mean(x, 0)
        # make prediction
        x = self.readout_layer(x)
        return x

    # training routine
    def train(self, training_data, test_data, nb_epochs, stopping_threshold, learning_rate, l1_reg, nb_reports=100):
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # Define loss function
        self.loss_func = torch.nn.MSELoss()
        report_epochs = nb_epochs / nb_reports
        for i in range(nb_epochs): # train for up to `nb_epochs` cycles
            loss = 0
            self.optimizer.zero_grad() # reset the gradients
            for datum in training_data:
                y_hat = self(datum) # make prediction
                loss += self.loss_func(y_hat, datum.y) # accumulate loss
            loss /= len(training_data) # normalize loss
            if loss.item() < stopping_threshold: # evaluate early stopping
                print("Breaking training loop at iteration {}\n".format(i))
                break
            if i % report_epochs == 0:
                print(f"Epoch\t{i}\t\t|\tLoss\t{loss}")
            loss += l1_reg * sum([torch.sum(abs(params)) for params in self.parameters()]) # L1 weight regularization
            loss.backward() # do back-propagation to get gradients
            self.optimizer.step() # update weights
        y_hat_test = torch.tensor([self(x) for x in test_data])
        y_test = torch.tensor([x.y for x in test_data])
        test_loss = self.loss_func(y_hat_test, y_test)
        print(f"\nTest loss: {test_loss}\n")