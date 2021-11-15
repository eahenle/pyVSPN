from numpy import concatenate
import torch
import torch_geometric


def choose_model(feature_length, args):
    """
    model = choose_model(feature_length, args)

    Returns the selected model according to `args` and `feature_length`
    """

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
    model = BondingGraphGNN(node_encoding_length, args)

    Creates a model for graph-level property prediction using MPNN on the bonding graph
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
    model = PoreGraphGNN(feature_length, args)

    An MPNN for making graph-level predictions on only the pore space network
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
    model = VSPN(feature_length, parallel, args)

    An MPNN for graph-level predictions using bond and pore graphs.
    If `parallel == True`, independent MPNNs process the two graphs and concatenate their graph-level readouts.
    If `parallel == False`, a single MPNN processes the combined bonding and pore input graph.
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
        av_embedding_length = args.av_embedding_length
        av_h = args.av_h
        av_mpnn_steps = args.av_mpnn_steps
        av_mpnn_aggr = args.av_mpnn_aggr

        super(VSPN, self).__init__()

        self.parallel = parallel
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        if parallel:
            self.atom_input = torch.nn.Linear(node_feature_length, atom_embedding_length, bias=False)
            self.voro_input = torch.nn.Linear(node_feature_length, voro_embedding_length, bias=False)
            self.atom_mpnn = torch_geometric.nn.conv.GatedGraphConv(atom_h, atom_mpnn_steps, atom_mpnn_aggr)
            self.voro_mpnn = torch_geometric.nn.conv.GatedGraphConv(voro_h, voro_mpnn_steps, voro_mpnn_aggr)
            self.prediction_layer = torch.nn.Linear(voro_h + atom_h, 1)
        else:
            self.av_input = torch.nn.Linear(node_feature_length, av_embedding_length, bias=False)
            self.av_mpnn = torch_geometric.nn.conv.GatedGraphConv(av_h, av_mpnn_steps, av_mpnn_aggr)
            self.prediction_layer = torch.nn.Linear(av_h, 1)
    
    def forward(self, datum):
        if self.parallel: # parallel VSPN
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
            atom_x = torch_geometric.nn.global_mean_pool(atom_x, datum.batch)
            voro_x = torch_geometric.nn.global_mean_pool(voro_x, datum.batch) ## TODO consider if mean is best...
            # nonlinear activation
            atom_x = self.relu(atom_x)
            voro_x = self.relu(voro_x)
            # make prediction
            return self.prediction_layer(concatenate(atom_x, voro_x)).squeeze(1)
        
        else: # connected VSPN
            x = self.av_input(datum.av_x)
            x = self.tanh(x)
            x = self.av_mpnn(x)
            x = torch_geometric.nn.global_mean_pool(x, datum.batch) ## TODO consider if mean is best...
            x = self.relu(x)
            return self.prediction_layer(x).squeeze(1)
