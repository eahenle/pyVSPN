## TODO API for PyCall, at least for post-training tasks

import torch
from sklearn.model_selection import train_test_split

from model import Model
from data_handling import load_data
from argument_parsing import parse_args, print_args
from misc import choose_device


def main():
    # get command line args (or defaults) and print them
    args = parse_args()
    print_args(args)

    # select training device (CPU/GPU)
    device = choose_device(args.device)

    # load data [and send to GPU]
    data, feature_length = load_data(args.properties, args.target, device)

    # instantiate the model [and send to GPU]
    model = Model(feature_length, args.node_encoding, args.graph_encoding, args.mpnn_steps)
    model.to(device)

    # split the data
    training_data, test_data = train_test_split(data, test_size=args.test_prop) ## TODO cache

    # run the training loop
    model.train(training_data, test_data, args.max_epochs, args.stop_threshold, args.learning_rate, args.l1_reg, args.nb_reports)


if __name__ == "__main__":
    main()
