import sys
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col, translate
from torch import nn
import torch

sys.path.append('/kaggle_simulations/agent/')

ROW = 7
COLUMN = 11

ROW_CENTER = int((ROW - 1) / 2)
COLUMN_CENTER = int((COLUMN - 1) / 2)


class Grid:
    FOOD = -1
    EMPTY = 0
    GOOSE_HEAD = 1
    GOOSE_BODY = 2
    GOOSE_TAIL = 3

    OTHER_HEAD = 4
    OTHER_TAIL = 5


NUM2ACTION = {
    0: "EAST",
    1: "SOUTH",
    2: "WEST",
    3: "NORTH",
}


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


def encode_observation(observation, action="NORTH"):
    board = np.zeros((ROW, COLUMN))
    for pos in observation["food"]:
        board[row_col(pos, COLUMN)] = Grid.FOOD

    for geese in observation["geese"]:
        for pos in geese:
            board[row_col(pos, COLUMN)] = Grid.GOOSE_BODY
        if len(geese) > 0:
            board[row_col(geese[-1], COLUMN)] = Grid.OTHER_TAIL
            board[row_col(geese[0], COLUMN)] = Grid.OTHER_HEAD

    i = observation["index"]
    self_goose = observation["geese"][i]
    if len(self_goose) > 0:
        board[row_col(self_goose[-1], COLUMN)] = Grid.GOOSE_TAIL
        board[
            row_col(translate(self_goose[0], Action[action].opposite(), COLUMN, ROW), COLUMN)
        ] = Grid.GOOSE_BODY  # virtual body to avoid taking opposite action
        head_pos = row_col(self_goose[0], COLUMN)
        board[head_pos] = Grid.GOOSE_HEAD
        board = np.roll(board, (ROW_CENTER - head_pos[0], COLUMN_CENTER - head_pos[1]), axis=(0, 1))

    return board


idx = 0
prev_action = "NORTH"

model = ConvDQN()
model.load_state_dict(torch.load('/kaggle_simulations/agent/model.pt', map_location=torch.device('cpu')))
model.eval()


def agent(obs_dict, config_dict):
    global idx, prev_action

    board = torch.from_numpy(encode_observation(obs_dict, action=prev_action)).float()
    action = model.greedy(board.unsqueeze(0))
    action = NUM2ACTION[action.squeeze().item()]

    prev_action = action
    return action
