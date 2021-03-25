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


def encode_env_stack_plus(env, interest_agent=(1, 1, 1, 1)):
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

    _, _, current_done_list, current_length_list = encode_state_stack_plus(env.steps[0])
    rollout_list = [list(), list(), list(), list()]
    split_rewards = [0] * 4
    for t in range(1, t_max):
        next_board_list, action_list, next_done_list, next_length_list = encode_state_stack_plus(env.steps[t])

        need_update_reward = 4
        update_agent = [1] * 4
        for i in range(num_agent):
            if not active_list[i]:
                update_agent[i] = -1
                need_update_reward -= 1
                continue
            if current_done_list[i] and next_done_list[i]:
                active_list[i] = False
                need_update_reward -= 1
                update_agent[i] = 0

        # if need_update_reward > 0:
        #     if need_update_reward == 4:
        #         pass
        #     if need_update_reward == 3:
        #         pass

        for i in range(num_agent):
            if not active_list[i]:
                continue

            reward = 0
            length_diff = next_length_list[i] - current_length_list[i]
            if t == t_max - 1:
                reward = final_rewards[i] + split_rewards[i]
            if length_diff < 0:
                reward = final_rewards[i] + split_rewards[i]
                
                if need_update_reward == 0 or need_update_reward == 1:
                    pass
                else:
                    for j in range(num_agent):
                        if update_agent[i] == 1:
                            split_rewards[i] += -reward/need_update_reward

            rollout_list[i].append((next_board_list[i], action_list[i], reward, next_done_list[i]))
        current_length_list = next_length_list
        current_done_list = next_done_list
    return rollout_list


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