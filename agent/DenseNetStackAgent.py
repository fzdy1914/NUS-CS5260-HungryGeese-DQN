import sys

import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate, Observation, adjacent_positions, \
    row_col
import torch

sys.path.append('/kaggle_simulations/agent/')
from dense_model import DenseNetStack
from silent_agent_helper import get_available_action

prev_action = "NORTH"

model = DenseNetStack()
model.load_state_dict(torch.load('/kaggle_simulations/agent/model_raw_100000.pt', map_location=torch.device('cpu')))
model.eval()


epoch = 0

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


STATUS = {
    "ACTIVE": 0,
    "DONE": 1
}

ACTION2NUM = {
    "NORTH": 0,
    "EAST": 1,
    "SOUTH": 2,
    "WEST": 3,
}

NUM2ACTION = {
    0: "NORTH",
    1: "EAST",
    2: "SOUTH",
    3: "WEST",
}


def encode_observation_stack(observation, action="NORTH", roll=False):
    board = np.zeros((4, ROW, COLUMN))
    food_board = np.zeros((1, ROW, COLUMN))
    for pos in observation["food"]:
        food_board[0][row_col(pos, COLUMN)] = Grid.FOOD

    for idx, geese in enumerate(observation["geese"]):
        for pos in geese:
            board[idx][row_col(pos, COLUMN)] = Grid.GOOSE_BODY
        if len(geese) > 0:
            board[idx][row_col(geese[-1], COLUMN)] = Grid.OTHER_TAIL
            board[idx][row_col(geese[0], COLUMN)] = Grid.OTHER_HEAD

    i = observation["index"]

    self_goose = observation["geese"][i]

    if len(self_goose) > 0:
        board[i][row_col(self_goose[-1], COLUMN)] = Grid.GOOSE_TAIL
        board[i][
            row_col(translate(self_goose[0], Action[action].opposite(), COLUMN, ROW), COLUMN)
        ] = Grid.GOOSE_BODY  # virtual body to avoid taking opposite action

        head_pos = row_col(self_goose[0], COLUMN)
        board[i][head_pos] = Grid.GOOSE_HEAD

        offset_x = ROW_CENTER - head_pos[0] if roll else 0
        offset_y = COLUMN_CENTER - head_pos[1] if roll else 0
        board = np.roll(board, (-i, offset_x, offset_y), axis=(0, 1, 2))
        board = np.concatenate([board, np.roll(food_board, (-i, offset_x, offset_y), axis=(0, 1, 2))], axis=0)
    else:
        return np.zeros((5, ROW, COLUMN))

    return board


def update(action):
    global prev_action, epoch
    epoch += 1
    prev_action = action


def agent(obs_dict, config_dict):
    global prev_action

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

    available_action = get_available_action(self_goose[0], other_goose_body, prev_action)

    board = torch.from_numpy(encode_observation_stack(obs_dict, action=prev_action, roll=False)).float()
    action_list = model.forward(board.unsqueeze(0)).squeeze().topk(k=4)

    ranked_action = list(filter(lambda x: x in available_action, [NUM2ACTION[i.item()] for i in action_list[1]]))
    if len(ranked_action) == 0:
        print("no way to go", epoch)
        action = NUM2ACTION[action_list[1][0].item()]
        update(action)
        return action

    action = ranked_action[0]

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
