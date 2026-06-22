import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = 3


class DuelingMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, ACTIONS)

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class DuelingCNN(nn.Module):
    def __init__(self, in_channels: int = 37):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, ACTIONS)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
        v = self.value(h)
        a = self.adv(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class DuelingC51CNN(nn.Module):
    def __init__(self, in_channels=37, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.n_atoms = n_atoms
        self.register_buffer(
            "support",
            torch.linspace(v_min, v_max, n_atoms)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
        )

        self.value = nn.Linear(256, n_atoms)
        self.adv = nn.Linear(256, ACTIONS * n_atoms)

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)

        v = self.value(h).view(-1, 1, self.n_atoms)
        a = self.adv(h).view(-1, ACTIONS, self.n_atoms)

        logits = v + (a - a.mean(dim=1, keepdim=True))
        return F.softmax(logits, dim=2)

    def expected_q(self, probs):
        return torch.sum(probs * self.support, dim=2)