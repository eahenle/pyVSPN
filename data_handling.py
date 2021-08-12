import numpy
import torch
import torch_geometric
import pandas
import random
import pickle
from tqdm import tqdm

from helper_functions import cached


class Dataset(torch_geometric.data.Dataset):
    """
    For loading mini-batches of data from disk. When iterated or sliced, returns a data lists corresponding to mini-batches,
    by loading the corresponding pickle files from the cache.
    """
    def __init__(self, batches, cache_path):
        self.batches = batches
        self.graph_cache = f"{cache_path}/graphs"

    # return number of batches
    def __len__(self):
        return len(self.batches)
    
    # return data list for given batch
    def __getitem__(self, batch_index):
        batch_names = self.batches[batch_index]
        batch = []
        for name in batch_names:
            with open(f"{self.graph_cache}/{name}.pkl", "rb") as f:
                batch.append(pickle.load(f))
        return batch


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

    # pack tensors as data object [and send to GPU]
    return torch_geometric.data.Data(x=x, edge_index=edge_index, y=y, batch=torch.tensor([0]), nb_nodes=x.shape[0], nb_edges=edge_index.shape[1])


def load_data(args, device):
    """
    Reads a collection of files from disk to build the data collection for working with the MPNN
    """
    # unpack args
    properties  = args.target_data
    target      = args.target
    input_path  = args.input_path
    cache_path  = args.cache_path

    # read the list of examples
    print("Reading example properties...")
    df = cached(lambda : pandas.read_csv(properties), "properties.pkl", args)
    feature_length = numpy.load(f"{input_path}/encoding_length.npy")

    # load graph arrays and pickle data objects for each graph
    names = [name for name in df["name"]]
    for i,name in enumerate(tqdm(df["name"], desc="Collecting Graph Data", mininterval=2)):
        cached(lambda : load_graph_arrays(name, df[target][i], input_path), f"graphs/{name}.pkl", args)

    # generate train/validate/test splits
    training_split, validation_split, test_split = cached(lambda : get_split_data(names, args), "data_split.pkl", args)

    # generate mini-batch splits
    training_data = cached(lambda : get_mini_batches(training_split, args), "minibatches.pkl", args)

    # load validation and testing data lists
    validation_data = load_data_list(validation_split, args)
    test_data = load_data_list(test_split, args)

    # set up training dataset
    training_data = Dataset(training_data, cache_path)

    return training_data, validation_data, test_data, feature_length


# splits data into training, validation, and test sets (shuffled name lists)
def get_split_data(names, args):
    assert len(names) > 0
    test_prop = args.test_prop

    # make shuffled list of data indices
    indices = [i for i,_ in enumerate(names)]
    random.shuffle(indices)

    # determine how many indices go into test and validation sets
    test_cut = int(test_prop * len(names))

    # cut the data into sets
    test_data = [names[i] for i in indices[0:test_cut]]
    validation_data = [names[i] for i in indices[test_cut:2*test_cut]]
    training_data = [names[i] for i in indices[2*test_cut:]]

    return training_data, validation_data, test_data


# splits list of indices into batches for mini-batch gradient descent
def get_mini_batches(data, args):
    batch_size = args.batch_size

    training_data = []
    while len(data) > 0: # keep going until all data indices are used
        batch = []
        for _ in range(batch_size): # build a list of between 1 and batch_size indices
            if len(data) > 0:
                batch.append(data.pop())
            else:
                break
        training_data.append(batch) # add batch to the list of batches

    return training_data


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
