import random

import torch
import torch.autograd as autograd
import torch.nn as nn


class ConvDQN(nn.Module):
    def __init__(self, feature_size=1344, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.dnn = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.dnn(x)
        return x

    def greedy(self, x):
        x = self.forward(x)
        x = x.max(dim=1)[1]
        return x

    def forward_max(self, x):
        x = self.forward(x)
        x = x.max(dim=1)[0]
        return x

    def act(self, x, epsilon=0.0):
        if random.random() < epsilon:
            return torch.randint(self.num_actions, size=(x.size(0),))
        else:
            return self.greedy(x)
