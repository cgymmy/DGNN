import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, num_layers: int = 4, input_size: int = 1, hidden_size: int = 128, output_size: int = 1, act="ReLU"):
        super(ResNet, self).__init__()
        activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
        }
        self.activation = activations[act]
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.activation,
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.input_layer(x)
        for layer in self.hidden_layers:
            y = y + self.activation(layer(y))
        return self.output_layer(y)
