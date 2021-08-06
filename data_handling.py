import numpy
import torch
import torch_geometric
import pandas
import sklearn

from misc import cached


class Dataset(torch_geometric.data.Dataset):
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
        return (datum.x, datum.edge_index, datum.batch, datum.y, datum.nb_nodes, datum.nb_edges)
    
    def __len__(self): # return the length of the data list
        return len(self.data_list)


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
    datum = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y, batch=torch.tensor([0]), nb_nodes=x.shape[0], nb_edges=edge_index.shape[1])
    return datum


def load_data(args):
    """
    Reads a collection of files from disk to build the data collection for working with the MPNN
    """
    # unpack args
    properties  = args.properties
    target      = args.target
    input_path  = args.input_path

    # read the list of examples
    print("Reading example properties...")
    df = pandas.read_csv(properties)
    # load in the data for each example
    names = df["name"]
    print("Loading data...")
    data = [load_graph_arrays(names[i], df[target][i], input_path) for i in range(len(df.index))]
    feature_length = numpy.load(f"{input_path}/encoding_length.npy")

    # split the data
    training_data, test_data = get_split_data(data, args)

    # generate training Dataset object
    training_data = Dataset(training_data)

    # split into mini-batches
    training_data = get_mini_batches(training_data, args)

    return training_data, test_data, feature_length


# splits and caches test/train data (or loads from cache file)
def get_split_data(data, args):
    f = lambda : sklearn.model_selection.train_test_split(data, test_size=args.test_prop)
    return cached(f, "data_split.pkl", args)


# splits and caches minibatches (or loads from cache file)
def get_mini_batches(training_data, args):
    f = lambda : torch_geometric.data.DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
    return cached(f, "minibatches.pkl", args)
