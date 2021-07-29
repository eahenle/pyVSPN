## TODO API for PyCall, at least for post-training tasks

import torch

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

    # model hyperparameters
    feature_length = 12 ## TODO record this in structure processing, pull in automatically

    # instantiate the model [and send to GPU]
    model = Model(feature_length, args.node_encoding, args.graph_encoding, args.mpnn_steps)
    model.to(device)

    # load data [and send to GPU]
    data = load_data(args.properties, args.target, device)

    # run the training loop
    model.train(data, args.max_epochs, args.stop_threshold, args.learning_rate, args.l1_reg, args.nb_reports)


if __name__ == "__main__":
    main()
