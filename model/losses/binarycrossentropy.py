import torch.nn.functional as F

class BinaryCrossEntropy:
    def __init__(self, parameters):
        
        self.parameters = parameters

    def __call__(self, x, y):
        log_prob = -1.0 * F.log_softmax(x, 1)
        loss = log_prob.gather(1, y.unsqueeze(1))
        loss = loss.mean()
        return loss