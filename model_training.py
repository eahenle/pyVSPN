import torch
import statistics
import numpy
from tqdm import tqdm

from helper_functions import save_checkpoint


# training routine
def train(model, training_data, validation_data, loss_func, args):
    assert len(training_data) > 0
    assert len(validation_data) > 0
    # unpack args
    nb_epochs = args.max_epochs
    learning_rate = args.learning_rate
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    nb_reports = args.nb_reports
    nb_checkpoints = args.nb_checkpoints
    output_path = args.output_path
    # get validation targets
    validation_y = torch.tensor([datum.y for datum in validation_data])
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    # open training log file
    with open(f"{output_path}/training_curve.csv", "w") as f:
        # write column headers
        f.write(f"Update,Training_MSE,Validation_MSE\n")
        # train for up to `nb_epochs` cycles
        updates = 0
        validation_loss_history = [numpy.inf, numpy.inf, numpy.inf]
        breakout = False
        for epoch_num in range(nb_epochs):
            training_loss = 0
            validation_loss = 0
            for batch in tqdm(training_data, desc=f"Training Epoch {epoch_num}"): # loop over minibatches
                # reset the gradients for the current mini-batch
                optimizer.zero_grad()
                # make predictions
                training_y_hat = torch.tensor([model(batch[j].x, batch[j].edge_index) for j in range(len(batch))])
                validation_y_hat = torch.tensor([model(datum.x, datum.edge_index) for datum in validation_data])
                # calculate batch loss and add to epoch training loss
                batch_loss = loss_func(training_y_hat, torch.tensor([datum.y for datum in batch]))
                training_loss += batch_loss / len(training_data)
                validation_loss = loss_func(validation_y, validation_y_hat)
                f.write(f"{updates},{training_loss},{validation_loss}\n")
                # check early stopping criteria
                stop, validation_loss_history = early_stopping(validation_loss.item(), validation_loss_history, args)
                if stop:
                    breakout = True
                    break
                # apply L1 regularization
                batch_loss += l1_reg * sum([torch.sum(abs(params)) for params in model.parameters()])
                batch_loss.backward() # do back-propagation to get gradients
                optimizer.step() # update weights
                updates += 1
        
            # print training reports
            if epoch_num % (nb_epochs // nb_reports) == 0:
                print(f"Epoch\t{epoch_num}\t\t|\tLoss\t{validation_loss}")
            
            if breakout:
                break
            
            # save model in case of training interruption
            if epoch_num % (nb_epochs // nb_checkpoints) == 0:
                save_checkpoint(model, args)


# evaluate early stopping
def early_stopping(validation_loss, validation_loss_history, args):
    stopping_threshold = args.stop_threshold
    stalling_threshold = args.stalling_threshold
    stop1 = False
    stop2 = False
    stop3 = False
    # stop training if validation loss gets low enough
    if validation_loss < stopping_threshold:
        stop1 = True
    # stop training if training loss increases several epochs in a row
    trues = 0
    for i in range(len(validation_loss_history)):
        if i == 0:
            continue
        if validation_loss_history[i] > validation_loss_history[i-1] and validation_loss > validation_loss_history[i]:
            trues += 1
        else:
            trues = 0
    if trues == len(validation_loss_history):
        stop2 = True
    # if % stdev over several epochs under threshold, end
    if statistics.pstdev(validation_loss_history) / validation_loss < stalling_threshold:
        stop3 = True
    
    validation_loss_history.pop(0)
    validation_loss_history.append(validation_loss)

    stop = any([stop1, stop2, stop3])

    return stop, validation_loss_history
