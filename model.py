import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


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
        x = self.greedy(x)
        for i in range(x.size(0)):
            if random.random() < epsilon:
                x[i] = torch.randint(self.num_actions, size=(1,))
        return x


class ConvDQNWithLength(nn.Module):
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
            nn.Linear(feature_size + 1, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x, length):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = torch.hstack((x, length))
        x = self.dnn(x)
        return x

    def greedy(self, x, length):
        x = self.forward(x, length)
        x = x.max(dim=1)[1]
        return x

    def forward_max(self, x, length):
        x = self.forward(x, length)
        x = x.max(dim=1)[0]
        return x

    def act(self, x, length, epsilon=0.0):
        x = self.greedy(x, length)
        for i in range(x.size(0)):
            if random.random() < epsilon:
                x[i] = torch.randint(self.num_actions, size=(1,))
        return x


class ConvD3QN(nn.Module):
    def __init__(self, feature_size=1344, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.action_dnn = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.state_dnn = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x_action, x_state = self.action_dnn(x), self.state_dnn(x)
        x = x_action + x_state.repeat(1, self.num_actions)
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
        x = self.greedy(x)
        for i in range(x.size(0)):
            if random.random() < epsilon:
                x[i] = torch.randint(self.num_actions, size=(1,))
        return x


class ConvD3QN_1(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
        )
        self.action_dnn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )
        self.state_dnn = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x_action, x_state = self.action_dnn(x), self.state_dnn(x)
        x = x_action + x_state.repeat(1, self.num_actions)
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
        x = self.greedy(x)
        for i in range(x.size(0)):
            if random.random() < epsilon:
                x[i] = torch.randint(self.num_actions, size=(1,))
        return x


class ConvD3QN_2(nn.Module):
    def __init__(self, feature_size=672, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
        )
        self.action_dnn = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.state_dnn = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x_action, x_state = self.action_dnn(x), self.state_dnn(x)
        x = x_action + x_state.repeat(1, self.num_actions)
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
        x = self.greedy(x)
        for i in range(x.size(0)):
            if random.random() < epsilon:
                x[i] = torch.randint(self.num_actions, size=(1,))
        return x


class ConvD3QN_3(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=2),
            nn.ReLU(),
        )
        self.action_dnn = nn.Sequential(
            nn.Linear(1440, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.state_dnn = nn.Sequential(
            nn.Linear(1440, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x_action, x_state = self.action_dnn(x), self.state_dnn(x)
        x = x_action + x_state.repeat(1, self.num_actions)
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
        x = self.greedy(x)
        for i in range(x.size(0)):
            if random.random() < epsilon:
                x[i] = torch.randint(self.num_actions, size=(1,))
        return x


class ConvD3QN_4(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2),
            nn.ReLU(),
        )
        self.action_dnn = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.state_dnn = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x_action, x_state = self.action_dnn(x), self.state_dnn(x)
        x = x_action + x_state.repeat(1, self.num_actions)
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
        x = self.greedy(x)
        for i in range(x.size(0)):
            if random.random() < epsilon:
                x[i] = torch.randint(self.num_actions, size=(1,))
        return x


class ConvD3QN_5d(nn.Module):
        def __init__(self, feature_size=1344, num_actions=4):
            super().__init__()
            self.num_actions = num_actions
            self.cnn = nn.Sequential(
                nn.Conv2d(5, 32, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3),
                nn.ReLU(),
            )
            self.action_dnn = nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
            )
            self.state_dnn = nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        def forward(self, x):
            if len(x.shape) == 3:
                x = x.view(x.size(0), 1, x.size(1), x.size(2))
            x = self.cnn(x)
            x = x.view(x.size(0), -1)
            x_action, x_state = self.action_dnn(x), self.state_dnn(x)
            x = x_action + x_state.repeat(1, self.num_actions)
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
            x = self.greedy(x)
            for i in range(x.size(0)):
                if random.random() < epsilon:
                    x[i] = torch.randint(self.num_actions, size=(1,))
            return x