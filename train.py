def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:  #assign tensors to device
        inputs, targets = input.to(device) 

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Loss: {loss.item()}')

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i+1}')
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print('-----------------------')
    print('Training is done')

"""
def do_train():
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)

    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")
    feed_forward_net = FeedForwardNet().to(device)
    print(feed_forward_net)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
"""