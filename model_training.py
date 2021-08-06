import torch
from tqdm import tqdm

from helper_functions import save_checkpoint


def un_pad_x(x, nb_nodes):
    return x[range(nb_nodes),:]


def un_pad_e(e, nb_edges):
    return e[:,range(nb_edges)]


# training routine
def train(model, training_data, validation_data, loss_func, args):
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
    # prep training curve record file
    with open(f"{output_path}/training_curve.csv", "w") as f:
            f.write(f"Epoch,Training_MSE,Validation_MSE\n")

    for i in tqdm(range(nb_epochs), desc="Training", mininterval=5): # train for up to `nb_epochs` cycles
        training_loss = 0
        for X,E,batch,y,nb_nodes,nb_edges in training_data: # loop over minibatches
            X = [un_pad_x(X[j], nb_nodes[j]) for j in range(len(nb_nodes))]
            E = [un_pad_e(E[j], nb_edges[j]) for j in range(len(nb_edges))]
            optimizer.zero_grad() # reset the gradients for the current batch
            y_hat = [model(X[j], E[j], batch[j]) for j in range(len(y))] # make predictions
            losses = [loss_func(y_hat[j], y[j]) for j in range(len(y))] # calculate losses
            batch_loss = sum(losses) / len(training_data) # accumulate normalized losses
            training_loss += batch_loss
            batch_loss += l1_reg * sum([torch.sum(abs(params)) for params in model.parameters()]) # L1 regularization
            batch_loss.backward() # do back-propagation to get gradients
            optimizer.step() # update weights

        # calculate epoch loss on validation set
        validation_y_hat = torch.tensor([model(datum.x, datum.edge_index, datum.batch) for datum in validation_data])
        epoch_loss = loss_func(validation_y, validation_y_hat)
        
        with open(f"{output_path}/training_curve.csv", "a") as f:
            f.write(f"{i},{training_loss/len(training_data)},{epoch_loss}\n")

        if early_stopping(epoch_loss.item(), args):
            break
        
        if i % (nb_epochs // nb_reports) == 0: # print training reports
            print(f"Epoch\t{i}\t\t|\tLoss\t{epoch_loss}")
        
        if i % (nb_epochs // nb_checkpoints) == 0: # save model in case of training interruption
            save_checkpoint(model, args)


# evaluate early stopping
def early_stopping(epoch_loss, args):
    stopping_threshold = args.stop_threshold
    if epoch_loss < stopping_threshold:
        return True
    else:
        return False
