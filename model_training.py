import torch

# training routine ## TODO implement checkpoints
def train(model, training_data, test_data, args):
    # unpack args
    nb_epochs = args.max_epochs
    stopping_threshold = args.stop_threshold
    learning_rate = args.learning_rate
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    nb_reports = args.nb_reports
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    # Define loss function
    loss_func = torch.nn.MSELoss()
    # train the model
    model.train()
    for i in range(nb_epochs): # train for up to `nb_epochs` cycles
        epoch_loss = 0
        
        for X,E,batch,y in training_data: # loop over minibatches
            batch_loss = 0
            optimizer.zero_grad() # reset the gradients for the current batch
            y_hat = [model(X[j], E[j], batch[j]) for j in range(len(y))] # make predictions
            losses = [loss_func(y_hat[j], y[j]) for j in range(len(y))] # calculate losses ## TODO diagnose tensor size warning
            batch_loss = sum(losses) / len(training_data) # accumulate normalized losses
            batch_loss += l1_reg * sum([torch.sum(abs(params)) for params in model.parameters()]) # L1 regularization
            batch_loss.backward() # do back-propagation to get gradients
            optimizer.step() # update weights
            epoch_loss += batch_loss ## TODO refactor to use a validation set for epoch loss
        
        if epoch_loss.item() < stopping_threshold: # evaluate early stopping ## TODO other early stopping criteria
            print(f"Breaking training loop at iteration {i}\n")
            break
        
        if i % (nb_epochs // nb_reports) == 0: # print training reports
            print(f"Epoch\t{i}\t\t|\tLoss\t{epoch_loss}") ## TODO record on disk for viz
        
    # evaluate test loss
    model.eval()
    y_hat_test = torch.tensor([model(datum.x, datum.edge_index, datum.batch) for datum in test_data])
    y_test = torch.tensor([x.y for x in test_data])
    test_loss = loss_func(y_hat_test, y_test)
    print(f"\nTest loss: {test_loss}\n")
