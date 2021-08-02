import torch

# training routine ## TODO implement checkpoints
def train(model, training_data, test_data, nb_epochs=1, stopping_threshold=0.1, learning_rate=0.01, l1_reg=0, l2_reg=0, nb_reports=1):
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    # Define loss function
    loss_func = torch.nn.MSELoss()
    report_epochs = nb_epochs / nb_reports
    for i in range(nb_epochs): # train for up to `nb_epochs` cycles ## TODO batch training
        loss = 0
        optimizer.zero_grad() # reset the gradients
        y_hat = [model(datum) for datum in training_data] # make predictions
        y = [datum.y for datum in training_data] # pull out target values
        losses = [loss_func(y_hat[j], y[j]) for j in range(len(y))] # calculate losses
        loss = sum(losses) / len(training_data) # accumulate normalized losses
        if loss.item() < stopping_threshold: # evaluate early stopping ## TODO other early stopping criteria
            print("Breaking training loop at iteration {}\n".format(i))
            break
        if i % report_epochs == 0:
            print(f"Epoch\t{i}\t\t|\tLoss\t{loss}") ## TODO record on disk for viz
        loss += l1_reg * sum([torch.sum(abs(params)) for params in model.parameters()]) # L1 regularization
        loss.backward() # do back-propagation to get gradients
        optimizer.step() # update weights
    y_hat_test = torch.tensor([model(x) for x in test_data])
    y_test = torch.tensor([x.y for x in test_data])
    test_loss = loss_func(y_hat_test, y_test)
    print(f"\nTest loss: {test_loss}\n")
