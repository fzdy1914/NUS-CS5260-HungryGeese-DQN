import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import translate

from parameters import *


def encode_state_stack_plus(state):
    board_list = list()
    action_list = list()
    reward_list = list()
    done_list = list()
    length_list = list()

    all_observation = state[0]['observation']

    board = np.zeros((16, 7 * 11))
    food_board = np.zeros((1, 7 * 11))

    for pos in all_observation["food"]:
        food_board[0, pos] = 1

    for idx, goose in enumerate(all_observation["geese"]):
        action = state[idx]["action"]
        if len(goose) > 0:
            board[idx * 4 + 0][goose[0]] = 1
            board[idx * 4 + 1][goose[-1]] = 1
            board[idx * 4 + 2][translate(goose[0], Action[action].opposite(), COLUMN, ROW)] = 1
        for pos in goose:
            board[idx * 4 + 3][pos] = 1

        length_list.append(len(goose))
        action_list.append(ACTION2NUM[action])
        reward_list.append(state[idx]["reward"])
        done_list.append(STATUS[state[idx]["status"]])

    for i in range(4):
        self_board = np.concatenate([np.roll(board, (-i * 4, 0), axis=(0, 1)), food_board], axis=0).reshape(-1, 7, 11)
        board_list.append(self_board)

    return board_list, action_list, done_list, length_list


def encode_env_stack_plus(env, buffer,
                         interest_agent=(1, 0, 0, 0),
                         normal_reward=DEFAULT_NORMAL_REWARD,
                         food_reward=DEFAULT_FOOD_REWARD,
                         food_reward_reduction_start=6,
                         hit_reward=DEFAULT_HIT_REWARD,
                         mode="normal",
                         ):
    num_agent = len(env.state)
    t_max = len(env.steps)
    active_list = [True if active == 1 else False for active in interest_agent]

    current_board_list, _, current_done_list, current_length_list = encode_state_stack_plus(env.steps[0])
    for t in range(1, t_max):
        next_board_list, action_list, next_done_list, next_length_list = encode_state_stack_plus(env.steps[t])

        for i in range(num_agent):
            if not active_list[i]:
                continue
            if current_done_list[i]:
                active_list[i] = False

        for i in range(num_agent):
            if not active_list[i]:
                continue

            reward = normal_reward
            length_diff = next_length_list[i] - current_length_list[i]
            if length_diff > 0:
                if mode == "length":
                    if current_length_list[i] <= food_reward_reduction_start:
                        reward = food_reward
                    else:
                        reward = 1 / (current_length_list[i] - food_reward_reduction_start)
                else:
                    reward = food_reward

            elif length_diff < 0:
                reward = hit_reward

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
                       ):
    num_agent = len(env.state)
    t_max = len(env.steps)
    active_list = [True if active == 1 else False for active in interest_agent]

    current_board_list, _, current_done_list, current_length_list = encode_state_stack_plus(env.steps[0])

    failure_t_list = [t_max] * num_agent

    for t in range(1, t_max):
        for i in range(num_agent):
            _, _, next_done_list, _ = encode_state_stack_plus(env.steps[t])
            if next_done_list[i] and t < failure_t_list[i]:
                failure_t_list[i] = t

    for t in range(1, t_max):
        next_board_list, action_list, next_done_list, next_length_list = encode_state_stack_plus(env.steps[t])

        for i in range(num_agent):
            if not active_list[i]:
                continue
            if next_done_list[i] and current_done_list[i]:
                active_list[i] = False

        for i in range(num_agent):
            if not active_list[i] or t <= failure_t_list[i] - failure_cutoff:
                continue

            reward = normal_reward
            length_diff = next_length_list[i] - current_length_list[i]
            if length_diff > 0:
                if mode == "length":
                    if current_length_list[i] <= food_reward_reduction_start:
                        reward = food_reward
                    else:
                        reward = 1 / (current_length_list[i] - food_reward_reduction_start)
                else:
                    reward = food_reward

            elif length_diff < 0:
                reward = hit_reward

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


def encode_observation_stack_plus(observation_list):
    observation = observation_list[-1]
    board = np.zeros((16, 7 * 11))
    food_board = np.zeros((1, 7 * 11))

    for pos in observation["food"]:
        food_board[0, pos] = 1

    for idx, goose in enumerate(observation["geese"]):
        if len(goose) > 0:
            board[idx * 4 + 0][goose[0]] = 1
            board[idx * 4 + 1][goose[-1]] = 1
            if len(observation_list) > 1:
                board[idx * 4 + 2][observation_list[-2]["geese"][idx][0]] = 1
        for pos in goose:
            board[idx * 4 + 3][pos] = 1

    i = observation["index"]
    board = np.concatenate([np.roll(board, (-i * 4, 0), axis=(0, 1)), food_board], axis=0).reshape(-1, 7, 11)

    return board