import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tqdm import tqdm

# train(model, train_set, test_set)
# parameters : epoch=2, log_interval, batch_size

class Train():
    def __init__(self, parameters):
        self.epochs = parameters['EPOCHS']
        self.batch_size = parameters['BATCH_SIZE']
        self.log_interval = parameters['LOG_INTERVAL']
        self.device = self.set_device()
        self.num_workers = self.set_num_workers()
        self.pin_memory = self.set_pin_memory()

    def set_device(self):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        return self.device

    def set_num_workers(self):
        if self.device == "cuda":
            return 1
        else:
            return 0

    def set_pin_memory(self):
        if self.device == "cuda":
            return True
        else:
            return False

    def number_of_correct(self, pred, y):
        # count number of correct predictions
        return pred.squeeze().eq(y).sum().item()

    def get_likely_index(self, tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    def train_epoch(self, model, train_loader, epoch, pbar, pbar_update):
        model.architeture.train()

        losses = []
        accuracy = []

        for batch_idx, (X, y) in enumerate(train_loader):

            X = X.to(self.device)
            y = y.to(self.device)

            # apply model on whole batch directly on device
            output = model.architeture(X)

            # count number of correct predictions
            pred = self.get_likely_index(output)
            correct = self.number_of_correct(pred, y)
            accuracy.append(correct/len(X))

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = model.loss(output.squeeze(), y)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            # print training stats
            if batch_idx % self.log_interval == 0:
                #print(f"Train Epoch: {epoch} [{batch_idx * len(X)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                s = "Train Epoch: {epoch} [{num_trained}/{len_dataset} ({perc_trained:.0f}%)]\tLoss: {loss:.4f}\tAcc: {num_correct}/{len_batch} ({perc_correct:.0f}%)\n"
                d = {
                    'epoch': epoch,
                    'num_trained': batch_idx * len(X),
                    'len_dataset': len(train_loader.dataset),
                    'perc_trained': 100. * batch_idx / len(train_loader),
                    'loss': loss.item(),
                    'num_correct': correct,
                    'len_batch': len(X),
                    'perc_correct': 100. * correct / len(X)
                }
                print(s.format(**d))
                #print(f"Train Epoch: {epoch} [{batch_idx * len(X)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

            # update progress bar
            pbar.update(pbar_update)
            # record loss
            losses.append(loss.item())

        return losses, accuracy

    def test_epoch(self, model, test_loader, epoch, pbar, pbar_update):
        model.architecture.eval()
        correct = 0
        losses = []

        for X, y in test_loader:

            X = X.to(self.device)
            y = y.to(self.device)

            # apply transform and model on whole batch directly on device
            #X = transform(X)
            output = model.architecture(X)

            # count number of correct predictions
            pred = self.get_likely_index(output)
            correct += self.number_of_correct(pred, y)

            # update progress bar
            pbar.update(pbar_update)

            # save losses 
            losses.append(model.loss(output.squeeze(), y).item())

        #print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
        s = "Test Epoch: {epoch} \tLoss: {loss:.4f}\tAcc: {num_correct}/{len_dataset} ({perc_correct:.0f}%)\n"
        d = {
            'epoch': epoch,
            'loss': model.loss.item(),
            'num_correct': correct,
            'len_dataset': len(test_loader.dataset),
            'perc_correct': 100. * correct / len(test_loader.dataset)
        }
        print(s.format(**d))

        accuracy = correct/len(test_loader.dataset)

        return losses, accuracy

    def train(self, model, train_set, test_set):
        
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        pbar_update = 1 / (len(train_loader) + len(test_loader))
        train_losses = {}
        train_accuracies = {}
        test_losses = {}
        test_accuracies = {}

        with tqdm(total=self.epochs) as pbar:
            for epoch in range(1, self.epochs + 1):
                train_losses[epoch], train_accuracies[epoch] = self.train_epoch(model, train_loader, epoch, pbar, pbar_update)
                test_losses[epoch], test_accuracies[epoch] = self.test_epoch(model, test_loader, epoch, pbar, pbar_update)
                model.scheduler.step()
        
        return train_losses, train_accuracies, test_losses, test_accuracies




        # Let's plot the training loss versus the number of iteration.
        # plt.plot(losses);
        # plt.title("training loss");
'''T = Train()
print(T.set_device())'''