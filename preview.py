#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os
import pandas
import sys

dd = sys.argv[1]

for exp_code in os.listdir(dd):
    output_path = f"{dd}/{exp_code}"

    training_curves = pandas.read_csv(f"{output_path}/training_curve.csv").sort_values("Epoch")

    epoch = training_curves["Epoch"]
    validation_loss = training_curves["Validation_MSE"]
    training_loss = training_curves["Training_MSE"]

    fig, ax = plt.subplots()
    plt.plot(epoch, validation_loss, label="Validation")
    plt.plot(epoch, training_loss, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Loss Curve Preview: {output_path}")
    plt.legend()
    plt.savefig(f"{output_path}/training_curve_preview.png")
