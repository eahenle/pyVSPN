import torch
import matplotlib.pyplot as plt
import pandas

def evaluate(model, test_data, loss_func, args):
    output_path = args.output_path

    # evaluate test loss
    y_hat_test = [model(datum).item() for datum in test_data]
    y_test = [x.y.item() for x in test_data]
    test_loss = loss_func(torch.tensor(y_hat_test), torch.tensor(y_test))
    print(f"\nTest loss: {test_loss}\n")

    # generate accuracy plot
    fig, ax = plt.subplots()
    im = plt.hist2d(y_test, y_hat_test, bins=20)
    plt.xlabel("Target Value (mmol/g)")
    plt.ylabel("Predicted Value (mmol/g)")
    plt.title(f"Test Accuracy (MSE = {test_loss})")
    fig.colorbar(im[3])
    plt.savefig(f"{output_path}/test_accuracy.png")

    # generate training curve
    training_curves = pandas.read_csv(f"{output_path}/training_curve.csv")
    update = training_curves["Update"]
    validation_loss = training_curves["Validation_MSE"]
    training_loss = training_curves["Training_MSE"]
    fig, ax = plt.subplots()
    plt.plot(update, validation_loss, label="Validation")
    plt.plot(update, training_loss, label="Training")
    #plt.xlim([0, ax.get_xlim()[1]]) ## TODO reinstate
    #plt.ylim([0, ax.get_ylim()[1]])
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig(f"{output_path}/training_curve.png")

