import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Autoencoder neural network to learn document representation.

"""


class Autoencoder(nn.Module):
    def __init__(self, layer_size, drop_prob=0.6):
        super().__init__()
        self.layer_size = layer_size
        self.fc = nn.ModuleList([])
        for i in range(len(layer_size) - 1):
            self.fc.append(
                nn.Conv2d(layer_size[i], layer_size[i + 1], (1, 5 if i == 0 else 1))
            )
        self.dropout = nn.Dropout(p=drop_prob, inplace=False)

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.dropout(F.relu(self.fc[i](x)))
        x = torch.mean(x, 3, keepdim=True)
        return x
