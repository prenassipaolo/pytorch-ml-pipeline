from torch import nn


class FeedForwardNet(nn.Module):

    def __init__(self, in_dim=1, hidden_dim=1, out_dim=1):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        x = self.dense_layers(x)
        x = self.softmax(x)
        return x