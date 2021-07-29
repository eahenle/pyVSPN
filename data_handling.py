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
    datum = torch_geometric.data.Data(x = x, edge_index = edge_index, y = y).to(device)
    return datum


# load data from disk
def load_data(properties, target, device):
    # read the list of examples
    df = pandas.read_csv(properties)
    # load in the data for each example
    names = df["name"]
    data = [load_graph_arrays(names[i], df[target][i], device) for i in range(len(df.index))]
    feature_length = numpy.load("encoding_length.npy")
    return data, feature_length
