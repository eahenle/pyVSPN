import numpy
import torch
from torch.nn import parallel
from torch.nn.modules.container import ModuleList
import torch_geometric
import pandas
import random
import pickle
from tqdm import tqdm
from torch_geometric.data import Data

from helper_functions import cached, _gaussian


# For packing pair of graphs
# ----------------------------------------------------------------------
class PairData(Data):
    '''
    It can be used to store pair of graphs in single DATA object.
    '''
    def __init__(self, edge_index_b=None, x_b=None, name_b = None, edge_index_v=None, x_v=None, name_v = None, y= None):
        super().__init__()
        self.edge_index_b = edge_index_b
        self.x_b = x_b
        self.name_b = name_b
        self.edge_index_v = edge_index_v
        self.x_v = x_v
        self.name_v = name_v
        self.y = y
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_b':
            return self.x_b.size(0)
        if key == 'edge_index_v':
            return self.x_v.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
# ----------------------------------------------------------------------

def load_graph_arrays(xtal_name, y, input_path, load_A, load_V, load_AV, encoding, args):
    # load targets into tensor
    y = torch.tensor([y], dtype=torch.float)

    # read the arrays encoding the graph
    if load_A:
        atom_edge_src = numpy.load(f"{input_path}/graphs/{xtal_name}_edges_src.npy")
        atom_edge_dst = numpy.load(f"{input_path}/graphs/{xtal_name}_edges_dst.npy")
        atom_node_fts = numpy.load(f"{input_path}/graphs/{xtal_name}_node_features.npy")
        # convert arrays to tensors
        atom_x = torch.tensor(atom_node_fts, dtype=torch.float)
        atom_edge_index = torch.tensor([atom_edge_src, atom_edge_dst], dtype=torch.long)

    if load_V:
        voro_edge_src = numpy.load(f"{input_path}/graphs/{xtal_name}_vspn_edges_src.npy")
        voro_edge_dst = numpy.load(f"{input_path}/graphs/{xtal_name}_vspn_edges_dst.npy")
        # load the radii of the voronoi spheres
        voro_node_rad = numpy.load(f"{input_path}/graphs/{xtal_name}_vspn_features.npy")
        # encode voro_node_rad
        voro_node_fts = encoding(voro_node_rad, args.voro_embedding)
        # convert arrays to tensors
        voro_x = torch.tensor(voro_node_fts, dtype=torch.float)
        voro_edge_index = torch.tensor([voro_edge_src, voro_edge_dst], dtype=torch.long)

    if load_AV:
        av_edge_src = numpy.load(f"{input_path}/graphs/{xtal_name}_av_edges_src.npy")
        av_edge_dst = numpy.load(f"{input_path}/graphs/{xtal_name}_av_edges_dst.npy")
        av_node_fts = numpy.load(f"{input_path}/graphs/{xtal_name}_av_features.npy")
        av_x = torch.tensor(av_node_fts, dtype=torch.float)
        av_edge_index = torch.tensor([av_edge_src, av_edge_dst], dtype=torch.long)

    # pack tensors into Data object
    if load_A and not load_V and not load_AV:
        data = torch_geometric.data.Data(x=atom_x, edge_index=atom_edge_index, y=y, name=xtal_name)
    elif load_V and not load_A and not load_AV:
        data = torch_geometric.data.Data(x=voro_x, edge_index=voro_edge_index, y=y, name=xtal_name)
    elif load_A and load_V and not load_AV:
        data = PairData(edge_index_b=atom_edge_index, x_b=atom_x, name_b = xtal_name, edge_index_v=voro_edge_index, x_v=voro_x, name_v=xtal_name, y=y)
    elif load_AV and not load_A and not load_V:
        data = torch_geometric.data.Data(x=av_x, edge_index=av_edge_index, y=y, name=xtal_name)
    else:
        assert(False, "Invalid model loading directives.")

    return data


def load_data(device, args):
    """
    Reads a collection of files from disk to build the data for working with the MPNN
    """
    # unpack args
    target_data = args.target_data
    target      = args.target
    input_path  = args.input_path
    batch_size  = args.batch_size
    model       = args.model
    sample      = args.sample
    
    load_A   = False
    load_V   = False
    load_AV  = False
    encoding = None

    if model == "ParallelVSPN":
        load_A = True
        load_V = True
        if args.encoding == "GaussianRBF":
            encoding = _gaussian
        else:
            raise Exception("Only Gaussian RBF is implemented (so far).")
    elif model == "JointVSPN":
        load_AV = True
    elif model == "BondingGraphGNN":
        load_A = True
    elif model == "PoreGraphGNN":
        load_V = True
        if args.encoding == "GaussianRBF":
            encoding = _gaussian
        else:
            raise Exception("Only Gaussian RBF is implemented (so far).")

    # read the list of examples
    df = cached(lambda : pandas.read_csv(target_data), "target_data.pkl", args)

    # load graph arrays and pickle data objects for each graph
    names = [name for name in df["name"]]
    targets = [target for target in df[target]]
    # apply random sampling
    if sample != 0:
        names = random.sample(names, sample)
        targets = random.sample(targets, sample)
    # track which name indices are skipped
    keep_i = numpy.ones(len(names))
    for i,name in enumerate(tqdm(names, desc="Collecting Graph Data", mininterval=2)):
        try:
            cached(lambda : load_graph_arrays(name, targets[i], input_path, load_A, load_V, load_AV, encoding, args), f"graphs/{name}.pkl", args)
        except Exception as exception:
            keep_i[i] = 0
            print(f"Warning: dropping {name}. Exception follows.\n{exception}")
    # filter names
    keeps = [i for i,x in enumerate(keep_i) if x == 1]
    names = [names[i] for i in keeps]
    # generate train/validate/test splits
    training_split, validation_split, test_split = cached(lambda : get_split_data(names, args), "data_split.pkl", args)

    # load data lists
    validation_data = load_data_list(validation_split, device, args)
    test_data = load_data_list(test_split, torch.device("cpu"), args)
    training_data = load_data_list(training_split, device, args)

    # printing sizes
    print(f'training_data_size: {len(training_data)}, validation_data_size: {len(validation_data)}, test_data_size: {len(test_data)}')
    # determine encoding length
    atom_feature_length = voro_feature_length = None
    if model == "BondingGraphGNN":
        atom_feature_length = training_data[0]["x"].shape[1]
    elif model == "PoreGraphGNN":
        voro_feature_length = training_data[0]["x"].shape[1]
    elif model == "ParallelVSPN":
        atom_feature_length = training_data[0]["x_b"].shape[1]
        voro_feature_length = training_data[0]["x_v"].shape[1]
    elif model == "JointVSPN":
        raise Exception("Model not implemented (yet).")
    else:
        raise Exception("Invalid model loading directives.")



    # cast training/validation data lists to DataLoader objects
    if load_A and load_V and not load_AV:
        training_data = torch_geometric.data.DataLoader(training_data, batch_size=batch_size, follow_batch=['x_b', 'x_v'], shuffle = True)
        validation_data = torch_geometric.data.DataLoader(validation_data, batch_size=len(validation_data), follow_batch=['x_b', 'x_v'])
        test_data = torch_geometric.data.DataLoader(test_data, batch_size=len(test_data), follow_batch=['x_b', 'x_v'])
    else:
        validation_data = torch_geometric.data.DataLoader(validation_data, batch_size=len(validation_data))
        training_data = torch_geometric.data.DataLoader(training_data, batch_size=batch_size, shuffle = True)
        test_data = torch_geometric.data.DataLoader(test_data, batch_size=len(test_data))

    return training_data, validation_data, test_data, atom_feature_length, voro_feature_length


# splits data into training, validation, and test sets (shuffled name lists)
def get_split_data(names, args):
    assert len(names) > 0
    test_prop = args.test_prop
    val_prop = args.val_prop
    assert test_prop + val_prop < 0.5

    # make shuffled list of data indices
    indices = [i for i,_ in enumerate(names)]
    random.shuffle(indices)

    # determine how many indices go into test and validation sets
    test_cut = int(test_prop * len(names))
    val_cut = int(val_prop * len(names))

    # cut the data into sets
    test_data = [names[i] for i in indices[0:test_cut]]
    validation_data = [names[i] for i in indices[test_cut:test_cut+val_cut]]
    training_data = [names[i] for i in indices[test_cut+val_cut:]]

    return training_data, validation_data, test_data


# loads a graph from the cache
def load_graph(name, cache_path):
    try:
        with open(f"{cache_path}/graphs/{name}.pkl", "rb") as f:
            graph = pickle.load(f)
        return graph
    except:
        pass


# generates a data list
def load_data_list(names, device, args):
    assert len(names) > 0
    cache_path = args.cache_path
    return [load_graph(name, cache_path).to(device) for name in names if load_graph(name, cache_path) != None ]
