import numpy
import torch
import torch_geometric
import pandas
import sklearn
import os
import pickle


class Dataset(torch.utils.data.Dataset):
    """
    data_set = Dataset(data_list)

    Transforms a data list into a data set for processing into mini-batches via torch's `DataLoader`
    """

    def __init__(self, data_list):
        self.data_list = data_list
        # tensors must all be same shape in a "stack". Determine maximum node/edge counts for padding smaller inputs
        max_nodes = max([datum.x.shape[0] for datum in data_list])
        max_edges = max([datum.edge_index.shape[1] for datum in data_list])
        for i in range(len(data_list)): # loop over data list indices
            # pad feature matrices
            x = self.data_list[i].x
            pad = torch.nn.ZeroPad2d((0, 0, 0, max_nodes - x.shape[0]))
            self.data_list[i].x = pad(x)
            # pad edge lists
            edge_index = self.data_list[i].edge_index
            pad = torch.nn.ZeroPad2d((0, max_edges - edge_index.shape[1], 0, 0))
            self.data_list[i].edge_index = pad(edge_index)
    
    def __getitem__(self, index):
        datum = self.data_list[index]
        return (datum.x, datum.edge_index, datum.batch, datum.y)
    
    def __len__(self): # return the length of the data list
        return len(self.data_list)


# load the serialized array representations of a graph, and collate into a Data object
def load_graph_arrays(xtal_name, y, graph_folder):
    # read the arrays encoding the graph
    edge_src = numpy.load(f"{graph_folder}/{xtal_name}_edges_src.npy")
    edge_dst = numpy.load(f"{graph_folder}/{xtal_name}_edges_dst.npy")
    node_fts = numpy.load(f"{graph_folder}/{xtal_name}_node_features.npy")
    
    # convert arrays to tensors
    x = torch.tensor(node_fts, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y = torch.tensor([y], dtype=torch.float)

    # pack tensors as data object [and send to GPU]
    datum = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y, batch=torch.tensor([0]))
    return datum


def load_data(args, device):
    """
    data, x_len = load_data(properties, target, device)

    Reads a collection of files from disk to build the data collection for working with the MPNN

    Arguments
    - properties: a string naming the CSV file from which to read structure names and target values
    - target: a string naming the column containing the target values
    - device: a string identifying the device to use for computations

    External Inputs
    - .npy files in graphs/: containing the node features and adjacency list information
    - encoding_length.npy: containing the length of the node feature vectors

    Outputs
    - data: a list of pytorch Data objects, containing node features (data[i].x), adjacency lists (data[i].edge_index), and target values (data[i].y)
    - feature_length: the length of the node feature vectors
    """
    # unpack args
    properties      = args.properties
    target          = args.target
    cache_path      = args.cache_path
    test_prop       = args.test_prop
    recache         = args.recache
    graph_folder    = args.graph_folder
    batch_size      = args.batch_size
    enc_len_file    = args.enc_len_file

    # read the list of examples
    df = pandas.read_csv(properties)
    # load in the data for each example
    names = df["name"]
    data = [load_graph_arrays(names[i], df[target][i], graph_folder) for i in range(len(df.index))]
    feature_length = numpy.load(enc_len_file)

    # split the data
    training_data, test_data = get_split_data(data, cache_path, test_prop, recache)

    # generate training Dataset object
    training_data = Dataset(training_data)

    # split into mini-batches
    training_data = get_mini_batches(training_data, batch_size, cache_path, recache)

    return training_data, test_data, feature_length


# splits and caches test/train data (or loads from cache file)
def get_split_data(data, cache_path, test_prop, recache):
    cache_file = f"{cache_path}/data_split.pkl"
    if recache or not os.path.isfile(cache_file): # do the split and save the cache file
        training_data, test_data = sklearn.model_selection.train_test_split(data, test_size=test_prop)
        cache_file = open(cache_file, "wb")
        pickle.dump((training_data, test_data), cache_file)
        cache_file.close()
    else: # load from the cache file
        cache_file = open(cache_file, "rb")
        training_data, test_data = pickle.load(cache_file)
        cache_file.close()
    return training_data, test_data


# splits and caches minibatches (or loads from cache file)
def get_mini_batches(training_data, batch_size, cache_path, recache):
    cache_file = f"{cache_path}/minibatches.pkl"
    if recache or not os.path.isfile(cache_file): # make minibatches and save the cache file
        training_data = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
        cache_file = open(cache_file, "wb")
        pickle.dump(training_data, cache_file)
        cache_file.close()
    else: # load from the cache file
        cache_file = open(cache_file, "rb")
        training_data = pickle.load(cache_file)
        cache_file.close()
    return training_data
