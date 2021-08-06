import torch
import matplotlib.pyplot as plt
import pandas

def evaluate(model, test_data, loss_func):
    # evaluate test loss
    y_hat_test = [model(datum.x, datum.edge_index, datum.batch).item() for datum in test_data]
    y_test = [x.y.item() for x in test_data]
    test_loss = loss_func(torch.tensor(y_hat_test), torch.tensor(y_test))
    print(f"\nTest loss: {test_loss}\n")

    # generate accuracy plot
    fig = plt.figure()
    im = plt.hist2d(y_test, y_hat_test, bins=20, density=True) # 2D histogram
    plt.xlabel("Target Value (mmol/g)")
    plt.ylabel("Predicted Value (mmol/g)")
    plt.title(f"Test Accuracy (MSE = {test_loss})")
    #fig.colorbar(im) ## TODO colorbar
    #plt.plot(fig.get_xlim(), fig.get_ylim()) ## TODO plot parity line
    plt.savefig("test_accuracy.png") ## TODO flag for where to save

    # generate training curve
    training_loss = pandas.read_csv("training_curve.csv")
    epoch = training_loss["Epoch"]
    training_loss = training_loss["Validation_MSE"]
    plt.figure()
    plt.plot(epoch, training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Loss")
    plt.savefig("training_curve.png")

