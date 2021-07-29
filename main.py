import torch

from model import MPNN, Model
from data_handling import load_data


def main():
    # data source
    properties = "properties.csv"
    target = "working_capacity_vacuum_swing [mmol/g]"

    # model hyperparameters
    node_encoding_length = 12 ## TODO record this in structure processing, pull in automatically
    hidden_encoding_length = 20 ## TODO make rest of HPs CL args
    graph_encoding_length = 15
    mpnn_steps = 5
    nb_epochs = 10000 # maximum number of training epochs
    stopping_threshold = 0.1 # average loss threshold for breaking out of the training loop

    # instantiate the model
    model = Model(node_encoding_length, hidden_encoding_length, graph_encoding_length, mpnn_steps)

    # load data
    data = load_data(properties, target)

    # instantiate objects for auto-differentiation and optimization
    optimizer = torch.optim.Adam(model.parameters()) ## TODO stuff into model class, set learning rate, regularize
    loss_func = torch.nn.MSELoss()

    # see what the randomly initialized model predicts (should be very bad)
    print("Before training:")
    y_hat = model(data[1])
    print("Truth:\n{}".format(data[1].y.item()))
    print("Prediction:\n{}\n".format(y_hat.item()))

    # run the training loop
    model.train(data, optimizer, loss_func, nb_epochs, stopping_threshold)

    # see what the trained model predicts (should be very good)
    print("After training:")
    y_hat = model(data[1])
    print("Truth:\n{}".format(data[1].y.item()))
    print("Prediction:\n{}\n".format(y_hat.item()))


if __name__ == "__main__":
    main()
