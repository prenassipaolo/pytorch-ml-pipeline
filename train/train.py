import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

# we will train with a learning rate of 0.01, but we will use a scheduler to decrease it to 0.001 during training after 20 epochs.
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) 
    


class Model:
    def __init__(self):
        """
        Parameters
        ----------
        
        """
        self.architecture = None

        # training attributes
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.parameters = None
        self.earlystopping = None
        
        # pre-trained attributes
        self.weights = None
        

    def __call__(self, epoch_score, model, model_path='checkpoint.pth'):
        return

    def get_architecture(MODEL_NAME, MODEL_PATH, PARAMETERS_PATH):
        #   ritorna architettura (istanza della classe)
        return

def train(model, epoch, log_interval, train_loader, optimizer, scheduler, loss):
    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):

        X = X.to(device)
        y = y.to(device)

        # apply model on whole batch directly on device
        output = model(X)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(X)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


def test(model, epoch):
    model.eval()
    correct = 0
    for X, y in test_loader:

        X = X.to(device)
        y = y.to(device)

        # apply model on whole batch directly on device
        output = model(X)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, y)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")



log_interval = 20
n_epoch = 2

pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch)
        scheduler.step()

# Let's plot the training loss versus the number of iteration.
# plt.plot(losses);
# plt.title("training loss");










'''def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        features = data['features'].to(device)
        label = data['label'].to(device)
        outputs = model(features)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss'''

'''
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
'''