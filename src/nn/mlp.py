import torch.nn as nn
import torch.nn.functional as F

from nn.activations import Sine


class MLP(nn.Module):
    def __init__(self, num_layers: int = 5, input_size: int = 1, hidden_size: int = 50, output_size: int = 1, act="relu"):
        super().__init__()
        activations = {
            'tanh': F.tanh,
            'sigmoid': F.sigmoid,
            'gelu': F.gelu,
            'relu': F.relu,
            'elu': F.elu,
            'sine': Sine(),
        }
        self.activation = activations[act]
        self.net = nn.ModuleList()
        self.net.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.net.append(nn.Linear(hidden_size, hidden_size))
        self.net.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.net[:-1]:
            x = self.activation(layer(x))
        return self.net[-1](x)
