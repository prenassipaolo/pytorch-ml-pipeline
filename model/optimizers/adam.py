import torch.optim as optim

class Adam:
    def __init__(self, learning_rate=0.001):
        
        self.learning_rate = learning_rate

    def function(self, model_parameters):
        return optim.Adam(model_parameters, lr=self.learning_rate)
    
    def __call__(self, model_parameters):
        return self.function(model_parameters)