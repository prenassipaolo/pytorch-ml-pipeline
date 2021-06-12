# file to create custom loss optimizer

import torch

def custom_optimizer(neural_network, LEARNING_RATE):
    return torch.optim.Adam(neural_network.parameters(), lr=LEARNING_RATE)