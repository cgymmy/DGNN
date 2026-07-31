import torch
import torch.nn as nn


class ParallelLinear(nn.Module):
    def __init__(self, num_modules, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_modules, out_features, in_features))
        self.bias = nn.Parameter(torch.randn(num_modules, out_features))

    def forward(self, x):
        # x shape: [N, ..., in_features] (例如 [N, n, m, 2])
        original_shape = x.shape
        x_flat = x.reshape(original_shape[0], -1, original_shape[-1])  # [N, L, in_features]
        output = torch.einsum("noi,nli->nlo", self.weight, x_flat)  # [N, L, out_features]
        output += self.bias.unsqueeze(1)
        return output.reshape(*original_shape[:-1], -1)
