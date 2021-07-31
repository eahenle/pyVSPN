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
    def __init__(self, node_encoding_length, hidden_encoding_length, mpnn_steps, mpnn_aggr="mean"):
        super(Model, self).__init__()
        assert hidden_encoding_length >= node_encoding_length
        # ReLU for nonlinear activation steps
        self.relu = torch.nn.ReLU()
        # input layer transforms encoding to hidden encoding length 
        self.input_layer = torch.nn.Linear(node_encoding_length, hidden_encoding_length)
        # MPNN develops latent representation
        self.mpnn_layers = MPNN(mpnn_steps, mpnn_aggr)
        # readout layer reduces node encoding matrix to fixed-length vector
        self.readout_layer = torch.nn.Linear(hidden_encoding_length, hidden_encoding_length)
        # prediction layer returns prediction from graph encoding vector
        self.prediction_layer = torch.nn.Linear(hidden_encoding_length, 1)
    
    # forward-pass behavior
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # transform input
        x = self.input_layer(x)
        x = self.relu(x)
        # do message passing
        x, _ = self.mpnn_layers(x, edge_index)
        x = self.relu(x)
        # do readout
        x = self.readout_layer(x)
        x = torch.mean(x, 0) ## TODO replace w/ set2set
        x = self.relu(x)
        # make prediction
        x = self.prediction_layer(x)
        return x

    # training routine
    def train(self, training_data, test_data, nb_epochs=1, stopping_threshold=0.1, learning_rate=0.01, l1_reg=0, l2_reg=0, nb_reports=1):
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=l2_reg)
        # Define loss function
        self.loss_func = torch.nn.MSELoss()
        report_epochs = nb_epochs / nb_reports
        for i in range(nb_epochs): # train for up to `nb_epochs` cycles
            loss = 0
            self.optimizer.zero_grad() # reset the gradients
            for datum in training_data: ## TODO batch training, vectorization
                y_hat = self(datum) # make prediction
                loss += self.loss_func(y_hat, datum.y) # accumulate loss
            loss /= len(training_data) # normalize loss
            if loss.item() < stopping_threshold: # evaluate early stopping ## TODO other early stopping criteria
                print("Breaking training loop at iteration {}\n".format(i))
                break
            if i % report_epochs == 0:
                print(f"Epoch\t{i}\t\t|\tLoss\t{loss}") ## TODO record on disk for viz
            loss += l1_reg * sum([torch.sum(abs(params)) for params in self.parameters()]) # L1 weight regularization ## TODO library function? L2?
            loss.backward() # do back-propagation to get gradients
            self.optimizer.step() # update weights
        y_hat_test = torch.tensor([self(x) for x in test_data])
        y_test = torch.tensor([x.y for x in test_data])
        test_loss = self.loss_func(y_hat_test, y_test)
        print(f"\nTest loss: {test_loss}\n")
