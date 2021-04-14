from enum import Enum

import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate

from parameters import *


def encode_state_stack(state, roll=True):
    all_observation = state[0]['observation']

    board = np.zeros((4, ROW, COLUMN))
    food_board = np.zeros((1, ROW, COLUMN))
    for pos in all_observation["food"]:
        food_board[0][row_col(pos, COLUMN)] = Grid.FOOD

    for idx, geese in enumerate(all_observation["geese"]):
        for pos in geese:
            board[idx][row_col(pos, COLUMN)] = Grid.GOOSE_BODY
        if len(geese) > 0:
            board[idx][row_col(geese[-1], COLUMN)] = Grid.OTHER_TAIL
            board[idx][row_col(geese[0], COLUMN)] = Grid.OTHER_HEAD

    board_list = list()
    action_list = list()
    reward_list = list()
    done_list = list()

    length_list = list()

    for i in range(len(state)):
        self_board = board.copy()
        self_goose = all_observation["geese"][i]
        action = state[i]["action"]

        if len(self_goose) > 0:
            self_board[i][row_col(self_goose[-1], COLUMN)] = Grid.GOOSE_TAIL
            self_board[i][
                row_col(translate(self_goose[0], Action[action].opposite(), COLUMN, ROW), COLUMN)
            ] = Grid.GOOSE_BODY  # virtual body to avoid taking opposite action

            head_pos = row_col(self_goose[0], COLUMN)
            self_board[i][head_pos] = Grid.GOOSE_HEAD

            offset_x = ROW_CENTER - head_pos[0] if roll else 0
            offset_y = COLUMN_CENTER - head_pos[1] if roll else 0

            self_board = np.roll(self_board, (-i, offset_x, offset_y), axis=(0, 1, 2))

            self_board = np.concatenate([self_board, np.roll(food_board, (-i, offset_x, offset_y), axis=(0, 1, 2))], axis=0)
        else:
            self_board = np.zeros((5, ROW, COLUMN))

        board_list.append(self_board)
        action_list.append(ACTION2NUM[action])
        reward_list.append(state[i]["reward"])
        done_list.append(STATUS[state[i]["status"]])
        length_list.append(len(self_goose))

    return board_list, action_list, done_list, length_list


def encode_env_stack(env, buffer,
                     interest_agent=(1, 0, 0, 0),
                     normal_reward=DEFAULT_NORMAL_REWARD,
                     food_reward=DEFAULT_FOOD_REWARD,
                     food_reward_reduction_start=6,
                     hit_reward=DEFAULT_HIT_REWARD,
                     mode="normal",
                     roll=True,
                     ):
    num_agent = len(env.state)
    t_max = len(env.steps)
    active_list = [True if active == 1 else False for active in interest_agent]

    last_step = env.steps[-1]
    rewards = {state['observation']['index']: state['reward'] for state in last_step}
    final_rewards = [0] * 4
    for p, r in rewards.items():
        for pp, rr in rewards.items():
            if p != pp:
                if r > rr:
                    final_rewards[p] += 1 / 3
                elif r < rr:
                    final_rewards[p] -= 1 / 3

    current_board_list, _, current_done_list, current_length_list = encode_state_stack(env.steps[0], roll=roll)
    for t in range(1, t_max):
        next_board_list, action_list, next_done_list, next_length_list = encode_state_stack(env.steps[t], roll=roll)

        for i in range(num_agent):
            if not active_list[i]:
                continue
            if next_done_list[i] and current_done_list[i]:
                active_list[i] = False

        for i in range(num_agent):
            if not active_list[i]:
                continue

            reward = 0
            length_diff = next_length_list[i] - current_length_list[i]

            if t == t_max - 1:
                reward = final_rewards[i]
            if length_diff < 0:
                reward = final_rewards[i]

            if mode == "length":
                buffer.add(board=current_board_list[i],
                           action=action_list[i],
                           reward=reward,
                           next_board=next_board_list[i],
                           done=next_done_list[i],
                           length=current_length_list[i],
                           next_length=next_length_list[i])
            else:
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


def encode_failure_env(env, buffer,
                       interest_agent=(1, 0, 0, 0),
                       normal_reward=DEFAULT_NORMAL_REWARD,
                       food_reward=DEFAULT_FOOD_REWARD,
                       food_reward_reduction_start=6,
                       hit_reward=DEFAULT_HIT_REWARD,
                       failure_cutoff=5,
                       mode="normal",
                       roll=True):
    num_agent = len(env.state)
    t_max = len(env.steps)
    active_list = [True if active == 1 else False for active in interest_agent]

    last_step = env.steps[-1]
    rewards = {state['observation']['index']: state['reward'] for state in last_step}
    final_rewards = [0] * 4
    for p, r in rewards.items():
        for pp, rr in rewards.items():
            if p != pp:
                if r > rr:
                    final_rewards[p] += 1 / 3
                elif r < rr:
                    final_rewards[p] -= 1 / 3

    current_board_list, _, current_done_list, current_length_list = encode_state_stack(env.steps[0], roll=roll)

    failure_t_list = [t_max] * num_agent

    for t in range(1, t_max):
        for i in range(num_agent):
            _, _, next_done_list, _ = encode_state_stack(env.steps[t], roll=roll)
            if next_done_list[i] and t < failure_t_list[i]:
                failure_t_list[i] = t

    for t in range(1, t_max):
        next_board_list, action_list, next_done_list, next_length_list = encode_state_stack(env.steps[t])

        for i in range(num_agent):
            if not active_list[i]:
                continue
            if next_done_list[i] and current_done_list[i]:
                active_list[i] = False

        for i in range(num_agent):
            if not active_list[i] or t <= failure_t_list[i] - failure_cutoff:
                continue

            reward = 0
            length_diff = next_length_list[i] - current_length_list[i]

            if t == t_max - 1:
                reward = final_rewards[i]
            if length_diff < 0:
                reward = final_rewards[i]

            if mode == "length":
                buffer.add(board=current_board_list[i],
                           action=action_list[i],
                           reward=reward,
                           next_board=next_board_list[i],
                           done=next_done_list[i],
                           length=current_length_list[i],
                           next_length=next_length_list[i])
            else:
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


def encode_observation_stack(observation, action="NORTH", roll=True):
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
