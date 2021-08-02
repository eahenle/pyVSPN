## TODO API for PyCall, at least for post-training tasks

import sklearn
import os
import pickle

from model import Model
from model_training import train
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
    model = Model(feature_length, args.node_encoding, args.mpnn_steps, args.s2s_steps)
    model.to(device)

    # split the data ## TODO add arg for ignoring and overwriting cache
    data_split_file = "./data_split.pkl"
    if os.path.isfile(data_split_file):
        data_split_file = open(data_split_file, "rb")
        training_data, test_data = pickle.load(data_split_file)
        data_split_file.close()
    else:
        training_data, test_data = sklearn.model_selection.train_test_split(data, test_size=args.test_prop)
        data_split_file = open(data_split_file, "wb")
        pickle.dump((training_data, test_data), data_split_file)
        data_split_file.close()

    # run the training loop
    train(model, training_data, test_data, args.max_epochs, args.stop_threshold, args.learning_rate, args.l1_reg, args.l2_reg, args.nb_reports)

    ## TODO visualize training curves, test accuracy


if __name__ == "__main__":
    main()
