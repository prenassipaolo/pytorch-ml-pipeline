import torch.optim as optim

class Adam:
    def __init__(self, parameters):
        
        self.learning_rate = parameters["LEARNING_RATE"]

    def function(self, model_parameters):
        return optim.Adam(model_parameters, lr=self.learning_rate)
    
    def __call__(self, model_parameters):
        return self.function(model_parameters)