import sys
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col, translate, Observation, adjacent_positions
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

model = ConvDQNWithLength()
model.load_state_dict(torch.load('/kaggle_simulations/agent/model.pt', map_location=torch.device('cpu')))
model.eval()


def agent(obs_dict, config_dict):
    global idx, prev_action

    observation = Observation(obs_dict)

    board = torch.from_numpy(encode_observation(obs_dict, action=prev_action)).float()
    length = torch.tensor(len(observation.geese[observation.index])).view(1, 1)
    action_list = model.forward(board.unsqueeze(0), length).squeeze().topk(k=4)
    action = NUM2ACTION[action_list[1][0].item()]
    if Action[prev_action].opposite() == Action[action]:
        print("0. try opposite")
        print("0. prev:", action, "0.now:", NUM2ACTION[action_list[1][1].item()])
        action = NUM2ACTION[action_list[1][1].item()]

    self_goose = observation.geese[observation.index]
    food_list = observation.food

    other_goose_head = list()
    other_goose_body = list()
    for i in range(len(observation.geese)):
        if i == observation.index:
            if len(observation.geese[i]) > 2:
                other_goose_body.extend(observation.geese[i][1:-2])
            continue
        if len(observation.geese[i]) > 0:
            other_goose_head.append(observation.geese[i][0])
            other_goose_body.extend(observation.geese[i])

    print("other_body", other_goose_body)
    if len(self_goose) > 0:
        self_goose_head = self_goose[0]
        next_pos_idx = translate(self_goose_head, Action[action], COLUMN, ROW)
        if next_pos_idx in other_goose_body:
            print("try hit")
            for act in range(1, 4):
                temp_action = NUM2ACTION[action_list[1][act].item()]
                if Action[prev_action].opposite() == Action[temp_action]:
                    continue
                if translate(self_goose_head, Action[temp_action], COLUMN, ROW) not in other_goose_body:
                    print("1. prev:", action, "1.now:", temp_action)
                    action = temp_action
                    break

        if next_pos_idx in food_list:
            for idx in adjacent_positions(next_pos_idx, COLUMN, ROW):
                finish = False
                if idx in other_goose_head:
                    print("danger pos")
                    for act in range(1, 4):
                        temp_action = NUM2ACTION[action_list[1][act].item()]
                        if Action[prev_action].opposite() == Action[temp_action]:
                            continue
                        if translate(self_goose_head, Action[temp_action], COLUMN, ROW) not in other_goose_body:
                            print("2. prev:", action, "2.now:", temp_action)
                            action = temp_action
                            finish = True
                            break

                if finish:
                    break

    prev_action = action
    return action
