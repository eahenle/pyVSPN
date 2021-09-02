#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas

output_path = "output"

# generate training curve
training_curves = pandas.read_csv(f"{output_path}/training_curve.csv")
update = training_curves["Update"]
validation_loss = training_curves["Validation_MSE"]
training_loss = training_curves["Training_MSE"]
fig, ax = plt.subplots()
plt.plot(update, validation_loss, label="Validation")
plt.plot(update, training_loss, label="Training")
plt.xlabel("Update")
plt.ylabel("MSE")
plt.title("Loss Curves")
plt.legend()
plt.savefig(f"{output_path}/training_curve.png")