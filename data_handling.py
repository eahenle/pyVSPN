import numpy
import torch
import torch_geometric
import pandas
import random
import pickle
from tqdm import tqdm

from helper_functions import cached


# load the serialized array representations of a graph, and collate into a Data object
def load_graph_arrays(xtal_name, y, input_path):
    # read the arrays encoding the graph
    edge_src = numpy.load(f"{input_path}/graphs/{xtal_name}_edges_src.npy")
    edge_dst = numpy.load(f"{input_path}/graphs/{xtal_name}_edges_dst.npy")
    node_fts = numpy.load(f"{input_path}/graphs/{xtal_name}_node_features.npy")
    
    # convert arrays to tensors
    x = torch.tensor(node_fts, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y = torch.tensor([y], dtype=torch.float)

    # pack tensors into Data object
    return torch_geometric.data.Data(x=x, edge_index=edge_index, y=y) ## TODO add tracking of input name


def load_data(args):
    """
    Reads a collection of files from disk to build the data for working with the MPNN
    """
    # unpack args
    target_data = args.target_data
    target      = args.target
    input_path  = args.input_path
    batch_size  = args.batch_size

    # read the list of examples
    df = cached(lambda : pandas.read_csv(target_data), "target_data.pkl", args)

    # load graph arrays and pickle data objects for each graph
    names = [name for name in df["name"]]
    for i,name in enumerate(tqdm(df["name"], desc="Collecting Graph Data", mininterval=2)):
        cached(lambda : load_graph_arrays(name, df[target][i], input_path), f"graphs/{name}.pkl", args)

    # generate train/validate/test splits
    training_split, validation_split, test_split = cached(lambda : get_split_data(names, args), "data_split.pkl", args)

    # load data lists
    validation_data = load_data_list(validation_split, args)
    test_data = load_data_list(test_split, args)
    training_data = load_data_list(training_split, args)

    # determine encoding length
    feature_length = training_data[0]["x"].shape[1]

    # cast training/validation data lists to DataLoader objects
    validation_data = torch_geometric.data.DataLoader(validation_data, batch_size=len(validation_data))
    training_data = torch_geometric.data.DataLoader(training_data, batch_size=batch_size)
    test_data = torch_geometric.data.DataLoader(test_data, batch_size=len(test_data))

    return training_data, validation_data, test_data, feature_length


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
    with open(f"{cache_path}/graphs/{name}.pkl", "rb") as f:
        graph = pickle.load(f)
    return graph


# generates a data list
def load_data_list(names, args):
    assert len(names) > 0
    cache_path = args.cache_path
    return [load_graph(name, cache_path) for name in names]
