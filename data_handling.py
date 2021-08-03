import numpy
import torch
import torch_geometric
import pandas
import sklearn
import os
import pickle


# load the serialized array representations of a graph, collate into a Data object, and send to device
def load_graph_arrays(xtal_name, y, device):
    # read the arrays encoding the graph
    edge_src = numpy.load("graphs/{}_edges_src.npy".format(xtal_name))
    edge_dst = numpy.load("graphs/{}_edges_dst.npy".format(xtal_name))
    node_fts = numpy.load("graphs/{}_node_features.npy".format(xtal_name))
    
    # convert arrays to tensors
    x = torch.tensor(node_fts, dtype=torch.float)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y = torch.tensor([y], dtype=torch.float)

    # pack tensors as data object [and send to GPU]
    datum = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y, batch=torch.tensor([0])).to(device) ## TODO load mini-batches via torch dataloader
    return datum


def load_data(properties, target, device, data_split_file, test_prop, recache):
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
    # read the list of examples
    df = pandas.read_csv(properties)
    # load in the data for each example
    names = df["name"]
    data = [load_graph_arrays(names[i], df[target][i], device) for i in range(len(df.index))]
    feature_length = numpy.load("encoding_length.npy")

    # split the data
    training_data, test_data = get_split_data(data, data_split_file, test_prop, recache)
    return training_data, test_data, feature_length


def get_split_data(data, data_split_file, test_prop, recache):
    if recache or not os.path.isfile(data_split_file):
        training_data, test_data = sklearn.model_selection.train_test_split(data, test_size=test_prop)
        data_split_file = open(data_split_file, "wb")
        pickle.dump((training_data, test_data), data_split_file)
        data_split_file.close()
    else:
        data_split_file = open(data_split_file, "rb")
        training_data, test_data = pickle.load(data_split_file)
        data_split_file.close()
    return training_data, test_data
