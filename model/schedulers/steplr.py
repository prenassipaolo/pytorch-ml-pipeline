import torch.optim as optim

class StepLR:
    def __init__(self, step_size, gamma):
        
        self.step_size = step_size
        self.gamma = gamma

    def function(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
    
    def __call__(self, optimizer):
        return self.function(optimizer)