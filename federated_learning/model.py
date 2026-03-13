"""
model.py - Shared model definition (copy this file to both server and client PCs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    Simple MLP for MNIST digit classification.
    Input:  28x28 grayscale image (flattened to 784)
    Output: 10 class logits
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)          # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)              # Raw logits (CrossEntropyLoss handles softmax)
        return x
