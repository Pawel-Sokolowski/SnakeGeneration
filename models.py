# models.py
import torch
import torch.nn as nn

ACTIONS = 3


class DuelingMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
        )
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, ACTIONS)

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class DuelingCNN(nn.Module):
    def __init__(self, in_channels, grid_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_size, grid_size)
            conv_out = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU(True),
        )
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, ACTIONS)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
        v = self.value(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=1, keepdim=True))
