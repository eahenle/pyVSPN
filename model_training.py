import torch

# training routine ## TODO implement checkpoints
def train(model, training_data, test_data, nb_epochs=1, stopping_threshold=0.1, learning_rate=0.01, l1_reg=0, l2_reg=0, nb_reports=1):
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    # Define loss function
    loss_func = torch.nn.MSELoss()
    report_epochs = nb_epochs / nb_reports
    for i in range(nb_epochs): # train for up to `nb_epochs` cycles
        loss = 0
        optimizer.zero_grad() # reset the gradients
        for datum in training_data: ## TODO batch training, vectorization
            y_hat = model(datum) # make prediction
            loss += loss_func(y_hat, datum.y) # accumulate loss
        loss /= len(training_data) # normalize loss
        if loss.item() < stopping_threshold: # evaluate early stopping ## TODO other early stopping criteria
            print("Breaking training loop at iteration {}\n".format(i))
            break
        if i % report_epochs == 0:
            print(f"Epoch\t{i}\t\t|\tLoss\t{loss}") ## TODO record on disk for viz
        loss += l1_reg * sum([torch.sum(abs(params)) for params in model.parameters()]) # L1 weight regularization ## TODO library function? L2?
        loss.backward() # do back-propagation to get gradients
        optimizer.step() # update weights
    y_hat_test = torch.tensor([model(x) for x in test_data])
    y_test = torch.tensor([x.y for x in test_data])
    test_loss = loss_func(y_hat_test, y_test)
    print(f"\nTest loss: {test_loss}\n")
