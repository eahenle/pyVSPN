import torch
import numpy
from tqdm import tqdm

from helper_functions import save_model

# training routine
def train(model, training_data, validation_data, loss_func, args):
    # unpack args
    nb_epochs = args.max_epochs
    learning_rate = args.learning_rate
    output_path = args.output_path
    verbose = args.verbose
    lr_decay_gamma = args.lr_decay_gamma

    # variable to track lowest validation loss (for selecting best model)
    best_val_loss = numpy.Inf

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma, verbose=verbose and not args.lr_decay_gamma == 1)

    # write training log column headers
    with open(f"{output_path}/training_curve.csv", "w") as f:
        f.write(f"Epoch,Training_MSE,Validation_MSE\n")

    # train for up to `nb_epochs` cycles
    for epoch_num in tqdm(range(nb_epochs), desc="Training"):
        training_loss = 0
        nb_training_graphs = 0
        # train model on each minibatch
        model.train()
        for training_batch in training_data:
            # increment the training graph counter
            nb_training_graphs += training_batch.num_graphs

            # reset the gradients for the current mini-batch
            optimizer.zero_grad()

            # calculate loss
            loss = loss_func(model(training_batch), training_batch.y)
            training_loss += loss.item() * training_batch.num_graphs

            # back-propagate and update weights
            loss.backward()
            optimizer.step()
        
        # update the learning rate scheduler
        scheduler.step()
        
        # normalize the loss
        training_loss /= nb_training_graphs

        # validate the model
        model.eval()
        validation_loss = 0
        nb_validation_graphs = 0
        with torch.no_grad():
            for val_batch in validation_data:
                loss = loss_func(model(val_batch), val_batch.y)
                validation_loss += loss.item() * val_batch.num_graphs
                nb_validation_graphs += val_batch.num_graphs
        validation_loss /= nb_validation_graphs

        # add to training log
        with open(f"{output_path}/training_curve.csv", "a") as f:
            f.write(f"{epoch_num+1},{training_loss},{validation_loss}\n")

        # preserve optimal model
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            save_model(model, args)
