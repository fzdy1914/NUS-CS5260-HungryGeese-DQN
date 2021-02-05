from enum import Enum

import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate

from parameters import *


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


def encode_state(state):
    all_observation = state[0]['observation']

    board = np.zeros((ROW, COLUMN))
    for pos in all_observation["food"]:
        board[row_col(pos, COLUMN)] = Grid.FOOD

    for geese in all_observation["geese"]:
        for pos in geese:
            board[row_col(pos, COLUMN)] = Grid.GOOSE_BODY
        if len(geese) > 0:
            board[row_col(geese[-1], COLUMN)] = Grid.OTHER_TAIL
            board[row_col(geese[0], COLUMN)] = Grid.OTHER_HEAD

    board_list = list()
    action_list = list()
    reward_list = list()
    done_list = list()

    length_list = list()

    for i in range(len(state)):
        self_board = board.copy()
        assert i == state[i]["observation"]["index"]
        self_goose = all_observation["geese"][i]
        action = state[i]["action"]
        if len(self_goose) > 0:
            self_board[row_col(self_goose[-1], COLUMN)] = Grid.GOOSE_TAIL
            self_board[
                row_col(translate(self_goose[0], Action[action].opposite(), COLUMN, ROW), COLUMN)
            ] = Grid.GOOSE_BODY  # virtual body to avoid taking opposite action
            head_pos = row_col(self_goose[0], COLUMN)
            self_board[head_pos] = Grid.GOOSE_HEAD
            self_board = np.roll(self_board, (ROW_CENTER - head_pos[0], COLUMN_CENTER - head_pos[1]), axis=(0, 1))

        board_list.append(self_board)
        action_list.append(ACTION2NUM[action])
        reward_list.append(state[i]["reward"])
        done_list.append(STATUS[state[i]["status"]])
        length_list.append(len(self_goose))

    return board_list, action_list, done_list, length_list


def encode_env(env, buffer, interest_agent=(1, 0, 0, 0)):
    num_agent = len(env.state)
    t_max = len(env.steps)
    active_list = [True if active == 1 else False for active in interest_agent]

    current_board_list, _, current_done_list, current_length_list = encode_state(env.steps[0])
    for t in range(1, t_max):
        next_board_list, action_list, next_done_list, next_length_list = encode_state(env.steps[t])

        for i in range(num_agent):
            if not active_list[i]:
                continue
            if next_done_list[i] and current_done_list[i]:
                active_list[i] = False

        for i in range(num_agent):
            if not active_list[i]:
                continue

            reward = N_REWARD
            length_diff = next_length_list[i] - current_length_list[i]
            if length_diff > 0:
                reward = 1
            elif length_diff < 0:
                reward = -10

            buffer.add(board=current_board_list[i],
                       action=action_list[i],
                       reward=reward,
                       next_board=next_board_list[i],
                       done=next_done_list[i])

            # print(current_board_list[i], action_list[i], reward, "\n", next_board_list[i], next_done_list[i], "\n")
        current_board_list = next_board_list
        current_length_list = next_length_list
        current_done_list = next_done_list
    buffer.on_episode_end()


if __name__ == '__main__':
    test_env = make("hungry_geese", debug=True)
    test_env.reset(4)
    test_env.run(['agent/wait_agent.py', 'agent/wait_agent.py', 'agent/wait_agent.py', 'agent/risk_averse_greedy.py'])

    from replay_buffer import buffer
    encode_env(test_env, buffer)
    print(buffer.sample(2))
