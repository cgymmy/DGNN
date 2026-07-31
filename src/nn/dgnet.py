import torch.nn as nn
import torch.nn.functional as F

from nn.layers import ParallelLinear


class DGNet(nn.Module):
    def __init__(self, num_modules, input_size: int = 2, hidden_size: int = 50, output_size: int = 1, num_layers: int = 2, act: str = 'tanh'):
        super().__init__()

        self.num_layers = num_layers
        self.activations = {
            'tanh': F.tanh,
            'sigmoid': F.sigmoid,
            'gelu': F.gelu,
            'relu': F.relu,
            'elu': F.elu,
        }

        if act not in self.activations:
            raise ValueError(f"Unsupported activation function: {act}")

        self.act = act
        self.layers = nn.ModuleList()
        self.layers.append(ParallelLinear(num_modules, input_size, hidden_size))
        for _ in range(num_layers - 1):
            self.layers.append(ParallelLinear(num_modules, hidden_size, hidden_size))
        self.layers.append(ParallelLinear(num_modules, hidden_size, output_size))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activations[self.act](x)
        x = self.layers[self.num_layers](x)
        return x
