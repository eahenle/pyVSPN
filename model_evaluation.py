import torch
import matplotlib.pyplot as plt
import pandas
import pickle
import numpy

def evaluate(model, test_data, loss_func, args):
    output_path = args.output_path
    target = args.target

    # evaluate test loss
    y_hat_test = [model(datum).item() for datum in test_data]
    y_test = [x.y.item() for x in test_data]
    test_loss = loss_func(torch.tensor(y_hat_test), torch.tensor(y_test))
    print(f"\nTest loss: {test_loss}\n")

    # generate accuracy plot
    fig, _ = plt.subplots()
    im = plt.hist2d(y_test, y_hat_test, bins=20)
    plt.xlabel("Target Value (mmol/g)")
    plt.ylabel("Predicted Value (mmol/g)")
    plt.title(f"Test Accuracy")
    fig.colorbar(im[3])
    plt.savefig(f"{output_path}/test_accuracy.png")

    # generate training curve
    avg_y = numpy.ones_like(y_test) * sum(y_test) / len(y_test)
    avg_y_mse = ((avg_y - y_test)**2).mean()
    training_curves = pandas.read_csv(f"{output_path}/training_curve.csv")
    update = training_curves["Epoch"]
    validation_loss = training_curves["Validation_MSE"]
    training_loss = training_curves["Training_MSE"]
    fig, _ = plt.subplots()
    plt.plot(update, validation_loss, label="Validation")
    plt.plot(update, training_loss, label="Training")
    plt.plot([0, len(update)-1], [avg_y_mse, avg_y_mse], label="Mean-Response", linestyle="dashed") # the MSE for always guessing the average on the test set
    plt.scatter([len(update)-1], [test_loss], label="Test", marker="*", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title(target)
    plt.legend()
    plt.savefig(f"{output_path}/training_curve.png")

