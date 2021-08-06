import torch
import matplotlib.pyplot as plt
import pandas

def evaluate(model, test_data, loss_func, args):
    # evaluate test loss
    y_hat_test = [model(datum.x, datum.edge_index, datum.batch).item() for datum in test_data]
    y_test = [x.y.item() for x in test_data]
    test_loss = loss_func(torch.tensor(y_hat_test), torch.tensor(y_test))
    print(f"\nTest loss: {test_loss}\n")

    # generate accuracy plot
    fig, ax = plt.subplots()
    im = plt.hist2d(y_test, y_hat_test, bins=20) # 2D histogram
    plt.xlabel("Target Value (mmol/g)")
    plt.ylabel("Predicted Value (mmol/g)")
    plt.title(f"Test Accuracy (MSE = {test_loss})")
    fig.colorbar(im[3]) ## TODO colorbar
    plt.plot([0, ax.get_xlim()[1]], [0, ax.get_ylim()[1]], c="white", linestyle="--")
    plt.savefig("test_accuracy.png") ## TODO flag for where to save

    # generate training curve
    training_curves = pandas.read_csv("training_curve.csv")
    epoch = training_curves["Epoch"]
    validation_loss = training_curves["Validation_MSE"]
    training_loss = training_curves["Training_MSE"]
    fig, ax = plt.subplots()
    plt.plot(epoch, validation_loss, label="Validation")
    plt.plot(epoch, training_loss, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("training_curve.png")

