import torch
import torch_geometric


def choose_model(atom_feature_length, voro_feature_length, args):
    """
    model = choose_model(feature_length, args)

    Returns the selected model according to `args` and `feature_length`
    """

    model = args.model

    if model == "BondingGraphGNN":
        return BondingGraphGNN(atom_feature_length, args)

    if model == "PoreGraphGNN":
        return PoreGraphGNN(voro_feature_length, args)

    if model == "ParallelVSPN":
        return VSPN(atom_feature_length, voro_feature_length, True, args)

    if model == "JointVSPN":
        return VSPN(feature_length, False, args)


class EmbeddingBlock(torch.nn.Module):
    '''
    A class to generate element embedding of nodes from one-hot encoding

    ...
    author: ali
    '''

    def __init__(self, feature_length, embedding_length, bias_flag=False):
        super().__init__()
        self.lin = torch.nn.Linear(feature_length, embedding_length, bias=bias_flag)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.lin(x))


class MPNNBlock(torch.nn.Module):
    '''

    A class containing mpnn and readout to generate graph level embedding.
    To be used after embedding block

    ...
    author: ali
    '''

    def __init__(self, hidden_size, mpnn_steps, mpnn_aggr, mpnn_readout):
        super().__init__()
        self.mpnn = torch_geometric.nn.conv.GatedGraphConv(hidden_size, mpnn_steps, mpnn_aggr)
        self.relu = torch.nn.ReLU()
        if mpnn_readout == "node_mean":
            self.readout = torch_geometric.nn.global_mean_pool
        else:
            raise Exception("Requested readout not recognized.")

    def forward(self, x, edge_index, batch):
        x = self.mpnn(x, edge_index)
        x = self.relu(x)
        x = self.readout(x, batch)
        return x


class OutputBlock(torch.nn.Module):
    '''
    A class containing a shallow network to generate final predictions
    To be used after mpnn block

    ...
    author: ali
    '''

    def __init__(self, hidden_size, bias_flag=True):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_size, hidden_size, bias=bias_flag)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(hidden_size, 1, bias=bias_flag)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        return self.softplus(self.lin2(self.relu(self.lin1(x))))


class BondingGraphGNN(torch.nn.Module):
    """
    Model to process bonding graphs.

    ...
    author = ali
    """

    # constructor
    def __init__(self, node_feature_length, args):
        super().__init__()
        # unpack args
        element_embedding_length = args.element_embedding
        atom_h = args.hidden_encoding
        mpnn_steps = args.mpnn_steps
        mpnn_aggr = args.mpnn_aggr


        # embedding block to transforms one-hot encoding
        self.emb_layer = EmbeddingBlock(feature_length = node_feature_length, embedding_length = element_embedding_length, bias_flag=False)
        # message passing and readout
        self.mpnn_layer = MPNNBlock(hidden_size = atom_h, mpnn_steps = mpnn_steps, mpnn_aggr = mpnn_aggr)
        # prediction layer returns prediction from graph encoding vector
        self.output_layer = OutputBlock(hidden_size = atom_h, bias_flag=True)

    # forward-pass behavior
    def forward(self, datum):
        # input embedding
        x = self.emb_layer(datum.x)
        # do message passing and readout
        x = self.mpnn_layer(x, datum.edge_index,datum.batch)
        # output block to make predictions
        x = self.output_layer(x)

        return x.squeeze(1)



class PoreGraphGNN(torch.nn.Module):
    """
    Model to process vornoi graphs.

    ...
    author = ali
    """

    def __init__(self, node_feature_length, args):
        super().__init__()
        #unpack args
        voro_h = args.voro_h
        vertex_embedding_length = args.voro_embedding
        mpnn_steps = args.mpnn_steps
        mpnn_aggr = args.mpnn_aggr


        # embedding block to transforms one-hot encoding
        self.emb_layer = EmbeddingBlock(feature_length = node_feature_length, embedding_length = vertex_embedding_length, bias_flag=False)
        # # message passing and readout
        self.mpnn_layer = MPNNBlock(hidden_size = voro_h, mpnn_steps = mpnn_steps, mpnn_aggr = mpnn_aggr)
        # prediction layer returns prediction from graph encoding vector
        self.output_layer = OutputBlock(hidden_size = voro_h, bias_flag=True)

    def forward(self, datum):
        # input embedding
        x = self.emb_layer(datum.x)
        # do message passing and readout
        x = self.mpnn_layer(x, datum.edge_index,datum.batch)
        # output block to make predictions
        x = self.output_layer(x)

        return x.squeeze(1)



class VSPN(torch.nn.Module):
    """
    Model to process both bonded and vornoi graphs

    ...
    author = ali
    """

    def __init__(self, atom_feature_length, voro_feature_length, parallel, args):
        super().__init__()
        #unpack args
        atom_embedding_length = args.element_embedding
        voro_embedding_length = args.voro_embedding
        voro_h = args.voro_h
        atom_h = args.hidden_encoding
        atom_mpnn_steps = args.mpnn_steps
        voro_mpnn_steps = args.mpnn_steps
        atom_mpnn_aggr = args.mpnn_aggr
        voro_mpnn_aggr = args.mpnn_aggr

        self.parallel = parallel ## TODO use a different class for JointVSPN

        if parallel:
            # embedding, mpnn, and readout for bonded graphs
            self.emb_layer_bond = EmbeddingBlock(feature_length = atom_feature_length, embedding_length = atom_embedding_length, bias_flag=False)
            self.mpnn_layer_bond = MPNNBlock(hidden_size = atom_h, mpnn_steps = atom_mpnn_steps, mpnn_aggr = atom_mpnn_aggr)
            # embedding, mpnn, and readout for vornoi graphs
            self.emb_layer_vor = EmbeddingBlock(feature_length = voro_feature_length, embedding_length = voro_embedding_length, bias_flag=False)
            self.mpnn_layer_vor = MPNNBlock(hidden_size = voro_h, mpnn_steps = voro_mpnn_steps, mpnn_aggr = voro_mpnn_aggr)

            # output layer to process concatenated embeddings of bonded and vornoi graphs
            self.output_layer = OutputBlock(hidden_size = voro_h + atom_h, bias_flag=True)
        else:
            pass

    def forward(self, datum):
        if self.parallel:  # parallel VSPN
            # unpacking args
            atom_x = datum.x_b
            atom_edge_index = datum.edge_index_b
            name_b = datum.name_b
            batch_b = datum.x_b_batch
            voro_x = datum.x_v
            voro_edge_index = datum.edge_index_v
            name_v = datum.name_v
            batch_v = datum.x_v_batch

            # embedding
            atom_x = self.emb_layer_bond(atom_x)
            voro_x = self.emb_layer_vor(voro_x)
            # mpnn and readout
            atom_x = self.mpnn_layer_bond(atom_x, atom_edge_index, batch_b)
            voro_x = self.mpnn_layer_bond(voro_x, voro_edge_index, batch_v)

            # make predictions
            return self.output_layer(torch.cat((atom_x, voro_x), dim=1)).squeeze(1)

        else:  # connected VSPN
            pass
