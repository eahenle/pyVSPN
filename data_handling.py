import numpy
import torch
import torch_geometric
import pandas


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
    datum = torch_geometric.data.Data(x = x, edge_index = edge_index, y = y, batch = torch.tensor([0])).to(device) ## TODO this can't be the right way to handle set2set
    return datum


def load_data(properties, target, device):
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
    return data, feature_length
