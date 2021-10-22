import torch

from model_definition import Model
from model_training import train
from data_handling import load_data
from argument_parsing import parse_args
from helper_functions import choose_device, check_paths, save_model, print_args, write_args
from model_evaluation import evaluate


def main():
    # get command line args (or defaults)
    args = parse_args()
    # display the args
    print_args(args)

    # check for required folders
    check_paths(args)

    # write the args as text in the output folder
    write_args(args)

    # select training device (CPU/GPU)
    device = choose_device(args)

    # load data as DataLoader objects
    training_data, validation_data, test_data, feature_length = load_data(args)

    # instantiate the model [and send to device]
    model = Model(feature_length, args).to(device)

    # Define loss function
    loss_func = torch.nn.L1Loss()

    # run the training loop
    model = train(model, training_data, validation_data, loss_func, args)

    # evaluate the model
    evaluate(model, test_data, loss_func, args)

    # save model
    save_model(model, args)


if __name__ == "__main__":
    main()
