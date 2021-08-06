import torch
import pickle


def un_pad_x(x, nb_nodes):
    return x[range(nb_nodes),:]


def un_pad_e(e, nb_edges):
    return e[:,range(nb_edges)]


# training routine
def train(model, training_data, validation_data, loss_func, args):
    # unpack args
    nb_epochs = args.max_epochs
    stopping_threshold = args.stop_threshold
    learning_rate = args.learning_rate
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    nb_reports = args.nb_reports
    # get validation targets
    validation_y = torch.tensor([datum.y for datum in validation_data])
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    # prep training curve record file
    with open("training_curve.csv", "w") as f:
            f.write(f"Epoch,Validation_MSE\n")

    for i in range(nb_epochs): # train for up to `nb_epochs` cycles
        for X,E,batch,y,nb_nodes,nb_edges in training_data: # loop over minibatches
            X = [un_pad_x(X[j], nb_nodes[j]) for j in range(len(nb_nodes))]
            E = [un_pad_e(E[j], nb_edges[j]) for j in range(len(nb_edges))]
            optimizer.zero_grad() # reset the gradients for the current batch
            y_hat = [model(X[j], E[j], batch[j]) for j in range(len(y))] # make predictions
            losses = [loss_func(y_hat[j], y[j]) for j in range(len(y))] # calculate losses
            batch_loss = sum(losses) / len(training_data) # accumulate normalized losses
            batch_loss += l1_reg * sum([torch.sum(abs(params)) for params in model.parameters()]) # L1 regularization
            batch_loss.backward() # do back-propagation to get gradients
            optimizer.step() # update weights

        validation_y_hat = torch.tensor([model(datum.x, datum.edge_index, datum.batch) for datum in validation_data])
        epoch_loss = loss_func(validation_y, validation_y_hat)
        
        with open("training_curve.csv", "a") as f:
            f.write(f"{i},{epoch_loss}\n")

        if epoch_loss.item() < stopping_threshold: # evaluate early stopping
            print(f"Breaking training loop at iteration {i}\n")
            break
        
        if i % (nb_epochs // nb_reports) == 0: # print training reports
            print(f"Epoch\t{i}\t\t|\tLoss\t{epoch_loss}")
        
        if i % (nb_epochs // nb_reports) == 0: # save model ## TODO add flag for checkpoint frequency
            with open("model_checkpoint.pkl", "wb") as f:
                pickle.dump(model, f)
