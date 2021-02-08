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


prev_prev_action = "NORTH"
prev_action = "NORTH"
epoch = 0

model = ConvD3QN()
model.load_state_dict(torch.load('/kaggle_simulations/agent/model.pt', map_location=torch.device('cpu')))
model.eval()


def can_rotate(goose):
    orders = [
        [Action.WEST, Action.NORTH, Action.EAST],
        [Action.SOUTH, Action.WEST, Action.NORTH],
        [Action.EAST, Action.SOUTH, Action.WEST],
        [Action.NORTH, Action.EAST, Action.SOUTH],

        [Action.EAST, Action.NORTH, Action.WEST],
        [Action.SOUTH, Action.EAST, Action.NORTH],
        [Action.WEST, Action.SOUTH, Action.EAST],
        [Action.NORTH, Action.WEST, Action.SOUTH],
    ]
    for order in orders:
        pos = goose[0]

        done = True
        for i in range(3):
            pos = translate(pos, order[i], COLUMN, ROW)
            if pos != goose[i + 1]:
                done = False
                break
        if done:
            return True, order[1].name
    return False, None


def get_available_action(pos, other_goose_body):
    action_list = list()
    for action in Action:
        if translate(pos, action, COLUMN, ROW) not in other_goose_body and action.opposite() != Action[prev_action]:
            action_list.append(action.name)
    return action_list


def get_adjacent_action(action):
    if action in ["NORTH", "SOUTH"]:
        return ["EAST", "WEST"]
    else:
        return ["NORTH", "SOUTH"]


def is_prev_same_direction():
    return prev_action == prev_prev_action


def update(action):
    global prev_prev_action, prev_action, epoch
    epoch += 1
    prev_prev_action = prev_action
    prev_action = action


def get_active_geese_num(goose):
    num = 0
    for geese in goose:
        if len(geese) > 0:
            num += 1
    return num


def agent(obs_dict, config_dict):
    global prev_action, epoch

    observation = Observation(obs_dict)
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
    # print("other_body", other_goose_body)

    available_action = get_available_action(self_goose[0], other_goose_body)

    board = torch.from_numpy(encode_observation(obs_dict, action=prev_action)).float()
    action_list = model.forward(board.unsqueeze(0)).squeeze().topk(k=4)

    ranked_action = list(filter(lambda x: x in available_action, [NUM2ACTION[i.item()] for i in action_list[1]]))
    if len(ranked_action) == 0:
        print("no way to go", epoch)
        action = NUM2ACTION[action_list[1][0].item()]
        update(action)
        return action

    # print(epoch, prev_action, ranked_action)

    action = ranked_action[0]

    if len(self_goose) == 4 and epoch < 128 and get_active_geese_num(observation.geese) > 2:
        ok, temp_action = can_rotate(self_goose)
        if ok:
            action = temp_action
            update(action)
            return action

        candidate_action = get_adjacent_action(prev_action)
        intersect_action = list(filter(lambda x: x in candidate_action, ranked_action))
        if len(intersect_action) > 0:
            action = intersect_action[0]
            if not is_prev_same_direction():
                if Action[prev_prev_action].opposite().name in intersect_action:
                    action = Action[prev_prev_action].opposite().name
                    print("Candidate", action)
            else:
                print("same")
            update(action)
            return action
        else:
            print("can not rotate")
    if len(self_goose) > 0:
        self_goose_head = self_goose[0]
        next_pos_idx = translate(self_goose_head, Action[action], COLUMN, ROW)
        if next_pos_idx in food_list:
            for idx in adjacent_positions(next_pos_idx, COLUMN, ROW):
                if idx in other_goose_head:
                    if len(ranked_action) > 1:
                        action = ranked_action[1]
                        break
                    else:
                        print("have to eat", ranked_action)
                    break

    update(action)
    return action
