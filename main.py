import torch

from model_definition import Model
from model_training import train
from data_handling import load_data
from argument_parsing import parse_args, print_args
from helper_functions import choose_device, check_paths, save_model
from model_evaluation import evaluate


def main():
    # get command line args (or defaults) and print them
    args = parse_args()
    print_args(args)

    # check for required folders
    check_paths(args)

    # select training device (CPU/GPU)
    device = choose_device(args)

    # load data [and send to device]
    training_data, test_data, feature_length = load_data(args) ## TODO validation data (currently using test data for validation)

    # instantiate the model [and send to device]
    model = Model(feature_length, args).to(device)

    # Define loss function
    loss_func = torch.nn.MSELoss()

    # run the training loop
    model.train() # set training mode
    train(model, training_data, test_data, loss_func, args)

    # evaluate the model
    model.eval() # set evaluation mode
    evaluate(model, test_data, loss_func)

    # save model
    save_model(model)


if __name__ == "__main__":
    main()
