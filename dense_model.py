import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3)
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        h = torch.cat([x[:, :, :, -1:], x, x[:, :, :, :1]], dim=3)
        h = torch.cat([h[:, :, -1:], h, h[:, :, :1]], dim=2)
        h = self.conv(h)
        h = self.bn(h)
        return h


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = ConvBlock(1, filters)
        self.blocks = nn.ModuleList([ConvBlock(filters, filters) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))

        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
        return p + v.repeat(1, 4)

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
                x[i] = torch.randint(4, size=(1,))
        return x


class DenseNetStack(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = ConvBlock(5, filters)
        self.blocks = nn.ModuleList([ConvBlock(filters, filters) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))

        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.head_p(h_head)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
        return p + v.repeat(1, 4)

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
                x[i] = torch.randint(4, size=(1,))
        return x

class DenseNetStackPlus(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 6, 32
        self.conv0 = ConvBlock(17, filters)
        self.blocks = nn.ModuleList([ConvBlock(filters, filters) for _ in range(layers)])
        self.action_dnn = nn.Linear(filters, 4, bias=False)
        self.state_dnn = nn.Linear(filters, 1, bias=False)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(x.size(0), 1, x.size(1), x.size(2))
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))

        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = self.action_dnn(h_avg)
        v = self.state_dnn(h_avg)
        return p + v.repeat(1, 4)

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
                x[i] = torch.randint(4, size=(1,))
        return x