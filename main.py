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
    training_data, test_data, feature_length = load_data(args.properties, args.target, device, args.data_split_file, args.test_prop, args.recache)

    # instantiate the model [and send to GPU]
    model = Model(feature_length, args.node_encoding, args.mpnn_steps, args.s2s_steps).to(device)

    # run the training loop
    train(model, training_data, test_data, args.max_epochs, args.stop_threshold, args.learning_rate, args.l1_reg, args.l2_reg, args.nb_reports)

    ## TODO visualize training curves, test accuracy


if __name__ == "__main__":
    main()
