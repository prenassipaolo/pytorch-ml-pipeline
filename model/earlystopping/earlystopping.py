# file to create custom early stopping

import numpy as np
import torch

class EarlyStopping:
    """
    A class used to save the best model before the train stops
    due to the early stopping

    ...

    Attributes
    ----------
    counter : int
        number of epochs after the checkpoint
    mode : str, optional
        defines if the criterion function should be maximum or minimum 
    best_score : float
        best criterion score (it correspond to the score of the last checkpoint)
    delta : float, optional
        improvement that the score has to obtain to continue the training
        it is still to implement
    patience : int
        number of epochs to run after the best checkpoint model before stopping
        if None the program should execute the training untit the last epoch
    checkpoint_path : string, optional
        Path where to save the model weights

    Methods
    -------
    __call__(epoch_score, model)
        Updates the class attributes where necessary (best_score, counter, early_stop)
        and saves the model if the new score is better
    save_checkpoint(self, epoch_score, model)
        Saves the model weights
    """

    def __init__(self, checkpoint_path='checkpoint.pth',  patience=np.inf, mode="max", delta=0.001):
        """
        Parameters
        ----------
        checkpoint_path : string, optional
            Path where to save the model weights
        patience : int, optional
            number of epochs to run after the best checkpoint model before stopping
            if None the program should execute the training untit the last epoch 
        mode : str, optional
            defines if the criterion function should be maximum or minimum 
        delta : float, optional
            improvement that the score has to obtain to continue the training
            it is still to implement
        """

        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.patience = np.min([patience, np.inf]) # if no value of patience is provided, set it to infinite
        self.checkpoint_path = checkpoint_path


    def __call__(self, epoch_score, model):

        """
        Updates the class attributes where necessary (best_score, counter, early_stop)
        and saves the model if the new score is better

        Parameters
        ----------
        epoch_score : float
            Score of the actual epoch
        model : torch.nn.Module
            Training model
        """

        if self.mode == "min":
            score = -1.0 * np.copy(epoch_score)
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model)
        elif score < self.best_score: #  + self.delta
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # ema.apply_shadow()
            self.save_checkpoint(epoch_score, model)
            # ema.restore()
            self.counter = 0

    def save_checkpoint(self, epoch_score, model):

        """
        Saves the model weights

        Parameters
        ----------
        epoch_score : float
            Score of the actual epoch
        model : torch.nn.Module
            Model to save

        Raises
        ------
        TypeError
            If the epoch score is degenerate.
        """


        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            # if not DEBUG:
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            raise TypeError("epoch_score in [-np.inf, np.inf, -np.nan, np.nan]")
