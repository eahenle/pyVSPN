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
    device = choose_device(args)

    # load data [and send to GPU]
    training_data, test_data, feature_length = load_data(args, device)

    # instantiate the model [and send to GPU]
    model = Model(feature_length, args).to(device)

    # run the training loop
    train(model, training_data, test_data, args)


if __name__ == "__main__":
    main()
