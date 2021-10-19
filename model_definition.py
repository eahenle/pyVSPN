import torch
import torch_geometric

class Model(torch.nn.Module):
    """
    model = Model(node_encoding_length, args)

    Creates a model for graph-level property prediction using an MPNN to develop node encodings,
    Set2Set for graph-level readout, and an affine prediction layer.
    """

    # constructor
    def __init__(self, node_feature_length, args):
        # unpack args
        element_embedding_length = args.element_embedding
        hidden_encoding_length = args.hidden_encoding
        mpnn_steps = args.mpnn_steps
        mpnn_aggr = args.mpnn_aggr

        # initialize base class
        super(Model, self).__init__()

        # ReLU for nonlinear activation
        self.activation = torch.nn.LeakyReLU()

        # input layer transforms encoding to hidden encoding length 
        self.input_layer = torch.nn.Linear(node_feature_length, element_embedding_length, bias=False)

        # MPNN develops latent representation
        self.mpnn = torch_geometric.nn.conv.GatedGraphConv(hidden_encoding_length, mpnn_steps, mpnn_aggr)

        # prediction layer returns prediction from graph encoding vector
        self.prediction_layer = torch.nn.Linear(hidden_encoding_length, 1)
    
    # forward-pass behavior
    def forward(self, datum):
        # embed input
        x = self.input_layer(datum.x)

        # do message passing
        x = self.mpnn(x, datum.edge_index)

        # nonlinear activation
        x = self.activation(x) # (element embedding)

        # do readout to graph-level encoding vector
        x = torch.mean(x, 0)
        
        # make prediction
        return self.prediction_layer(x)
