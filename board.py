from enum import Enum

import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate


class Grid:
    FOOD = -1
    EMPTY = 0
    GOOSE_HEAD = 1
    GOOSE_BODY = 2
    GOOSE_TAIL = 3


def encode_env(env, ROW, COLUMN):
    state = env.state
    all_observation = state[0]['observation']

    board = np.zeros((ROW, COLUMN))
    for pos in all_observation["food"]:
        board[row_col(pos, COLUMN)] = Grid.FOOD

    for geese in all_observation["geese"]:
        for pos in geese:
            board[row_col(pos, COLUMN)] = Grid.GOOSE_BODY

    board_list = list()
    reward_list = list()
    for i in range(len(state)):
        self_board = board.copy()
        assert i == state[i]["observation"]["index"]
        self_goose = all_observation["geese"][i]
        self_board[row_col(self_goose[-1], COLUMN)] = Grid.GOOSE_TAIL
        action = Action[state[i]["action"]]
        self_board[row_col(translate(self_goose[0], action.opposite(), COLUMN, ROW), COLUMN)] = Grid.GOOSE_BODY
        self_board[row_col(self_goose[0], COLUMN)] = Grid.GOOSE_HEAD
        board_list.append(self_board)
        reward_list.append(state[i]["reward"])
    return board_list, reward_list


if __name__ == '__main__':
    test_env = make("hungry_geese", debug=True)
    test_env.reset(4)
    COLUMN = test_env.configuration["columns"]
    ROW = test_env.configuration["rows"]
    encode_env(test_env, ROW, COLUMN)
