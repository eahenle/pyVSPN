from numpy import concatenate
import torch
import torch_geometric

def choose_model(feature_length, args):
    model = args.model

    if model == "BondingGraphGNN":
        return BondingGraphGNN(feature_length, args)
    if model == "PoreGraphGNN":
        return PoreGraphGNN(feature_length, args)
    if model == "ParallelVSPN":
        return VSPN(feature_length, True, args)
    if model == "JointVSPN":
        return VSPN(feature_length, False, args)

class BondingGraphGNN(torch.nn.Module):
    """
    model = Model(node_encoding_length, args)

    Creates a model for graph-level property prediction using an MPNN to develop node encodings,
    Set2Set for graph-level readout, and an affine prediction layer.
    """

    # constructor
    def __init__(self, node_feature_length, args):
        # unpack args
        element_embedding_length = args.element_embedding
        atom_h = args.hidden_encoding
        mpnn_steps = args.mpnn_steps
        mpnn_aggr = args.mpnn_aggr

        # initialize base class
        super(BondingGraphGNN, self).__init__()

        # nonlinear activations
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        # input layer transforms encoding to hidden encoding length 
        self.atom_input = torch.nn.Linear(node_feature_length, element_embedding_length, bias=False)

        # MPNN develops latent representation
        self.atom_mpnn = torch_geometric.nn.conv.GatedGraphConv(atom_h, mpnn_steps, mpnn_aggr)

        # prediction layer returns prediction from graph encoding vector
        self.prediction_layer = torch.nn.Linear(atom_h, 1)
    
    # forward-pass behavior
    def forward(self, datum):
        # embed input
        x = self.atom_input(datum.x)

        # nonlinear activation
        x = self.tanh(x)

        # do message passing
        x = self.atom_mpnn(x, datum.edge_index)

        # do readout to graph-level encoding vector
        #x = torch_scatter.scatter_mean(x, datum.batch)
        x = torch_geometric.nn.global_mean_pool(x, datum.batch)

        # nonlinear activation
        x = self.relu(x)
        
        # make prediction
        return self.prediction_layer(x).squeeze(1)


class PoreGraphGNN(torch.nn.Module):
    """
    """
    
    def __init__(self, node_feature_length, args):
        voro_h = args.voro_h
        vertex_embedding_length = args.voro_embedding
        mpnn_steps = args.mpnn_steps
        mpnn_aggr = args.mpnn_aggr

        super(PoreGraphGNN, self).__init__()

        self.voro_input = torch.nn.Linear(node_feature_length, vertex_embedding_length, bias=False)
        self.tanh = torch.nn.Tanh()
        self.mpnn = torch_geometric.nn.conv.GatedGraphConv(voro_h, mpnn_steps, mpnn_aggr)
        self.relu = torch.nn.ReLU()
        self.prediction_layer = torch.nn.Linear(voro_h)

    def forward(self, datum):
        # embed input
        x = self.voro_input(datum.voro_x)

        # nonlinear activation
        x = self.tanh(x)

        # do message passing
        x = self.mpnn(x, datum.edge_index)

        # do readout to graph-level encoding vector
        #x = torch_scatter.scatter_mean(x, datum.batch)
        x = torch_geometric.nn.global_mean_pool(x, datum.batch)

        # nonlinear activation
        x = self.relu(x)
        
        # make prediction
        return self.prediction_layer(x).squeeze(1)


class VSPN(torch.nn.Module):
    """
    """

    def __init__(self, node_feature_length, parallel, args):
        atom_embedding_length = args.element_embedding
        voro_embedding_length = args.vertex_embedding
        voro_h = args.voro_h
        atom_h = args.atom_h
        atom_mpnn_steps = args.atom_mpnn_steps
        voro_mpnn_steps = args.voro_mpnn_steps
        atom_mpnn_aggr = args.atom_mpnn_aggr
        voro_mpnn_aggr = args.voro_mpnn_aggr

        super(VSPN, self).__init__()

        self.atom_input = torch.nn.Linear(node_feature_length, atom_embedding_length, bias=False)
        self.voro_input = torch.nn.Linear(node_feature_length, voro_embedding_length, bias=False)
        self.tanh = torch.nn.Tanh()
        self.atom_mpnn = torch_geometric.nn.conv.GatedGraphConv(atom_h, atom_mpnn_steps, atom_mpnn_aggr)
        self.voro_mpnn = torch_geometric.nn.conv.GatedGraphConv(voro_h, voro_mpnn_steps, voro_mpnn_aggr)
        self.relu = torch.nn.ReLU()
        self.prediction_layer = torch.nn.Linear(voro_h + atom_h, 1)
    
    def forward(self, datum):
        # embed inputs
        atom_x = self.atom_input(datum.atom_x)
        voro_x = self.voro_input(datum.voro_x)

        # nonlinear activation
        atom_x = self.tanh(atom_x)
        voro_x = self.tanh(voro_x)

        # do message passing
        atom_x = self.atom_mpnn(atom_x, datum.atom_edge_index)
        voro_x = self.voro_mpnn(voro_x, datum.voro_edge_index)

        # do readout to graph-level encoding vector
        #x = torch_scatter.scatter_mean(x, datum.batch)
        atom_x = torch_geometric.nn.global_mean_pool(atom_x, datum.batch)
        voro_x = torch_geometric.nn.global_mean_pool(voro_x, datum.batch)

        # nonlinear activation
        atom_x = self.relu(atom_x)
        voro_x = self.relu(voro_x)
        
        # make prediction
        return self.prediction_layer(concatenate(atom_x, voro_x)).squeeze(1)
