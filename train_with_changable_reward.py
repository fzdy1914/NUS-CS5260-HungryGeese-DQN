import math

import torch

from multiprocessing import Process, Event, SimpleQueue
from kaggle_environments import make
import time
import numpy as np
from torch import optim
from tqdm import trange
from model import ConvDQNWithLength
from board import encode_state, encode_env
import torch.nn.functional as F

from cpprb import ReplayBuffer, MPPrioritizedReplayBuffer
from parameters import *


def compute_epsilon(episode, min_epsilon, max_epsilon, epsilon_decay):
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon


def unpack_sample(sample):
    board = torch.from_numpy(sample["board"]).float().cuda()
    action = torch.from_numpy(sample["action"]).to(torch.int64).cuda()
    reward = torch.from_numpy(sample["reward"]).float().cuda().squeeze()
    next_board = torch.from_numpy(sample["next_board"]).float().cuda()
    done = torch.from_numpy(sample["done"]).float().cuda().squeeze()
    length = torch.from_numpy(sample["length"]).float().cuda()
    next_length = torch.from_numpy(sample["next_length"]).float().cuda()
    return board, action, reward, next_board, done, length, next_length


def Q_model(model, board, action, length):
    return model.forward(board, length).gather(dim=1, index=action).squeeze()


def Q_target(model, board, reward, done, length):
    with torch.no_grad():
        x = reward + GAMMA * model.forward_max(board, length) * (1.0 - done)
        return x


def both_Q(model, target, unpacked_sample):
    board, action, reward, next_board, done, length, next_length = unpacked_sample
    return Q_model(model, board, action, length), Q_target(target, next_board, reward, done, next_length)


def abs_TD(model, target, sample):
    unpacked_sample = unpack_sample(sample)
    with torch.no_grad():
        q1, q2 = both_Q(model, target, unpacked_sample)
        return torch.abs(q1 - q2)


def explorer(global_rb, is_training_done, queue):
    local_rb = ReplayBuffer(LOCAL_BUFFER_SIZE, ENV_DICT_WITH_LENGTH)

    model = ConvDQNWithLength().cuda()
    target = ConvDQNWithLength().cuda()
    target.load_state_dict(model.state_dict())
    env = make("hungry_geese", debug=False)

    epsilon = 0
    while not is_training_done.is_set():
        if not queue.empty():
            model_state, target_state, epsilon = queue.get()
            model.load_state_dict(model_state)
            target.load_state_dict(target_state)

        env.reset(4)
        while not env.done:
            board_list, _, _, length_list = encode_state(env.state)
            board = torch.from_numpy(np.stack(board_list)).float().cuda()
            length = torch.from_numpy(np.stack(length_list)).float().cuda().view(len(length_list), 1)
            action = model.act(board, length, epsilon=epsilon)
            action = [NUM2ACTION[i.item()] for i in action]
            env.step(action)

        encode_env(env, local_rb, (1, 1, 1, 1), mode="length")

        if local_rb.get_stored_size() >= LOCAL_BUFFER_SIZE:
            sample = local_rb.get_all_transitions()
            TD = abs_TD(model, target, sample).detach().cpu().numpy()
            global_rb.add(**sample, priorities=TD)
            local_rb.clear()


if __name__ == "__main__":
    num_episode = 5000000
    min_epsilon, max_epsilon, epsilon_decay = 0, 0.1, 500000

    global_rb = MPPrioritizedReplayBuffer(BUFFER_SIZE, ENV_DICT_WITH_LENGTH)

    is_training_done = Event()
    is_training_done.clear()

    qs = [SimpleQueue() for _ in range(N_EXPLORER)]
    ps = [Process(target=explorer,
                  args=(global_rb, is_training_done, q))
          for q in qs]

    for p in ps:
        p.start()

    model = ConvDQNWithLength().cuda()
    model.load_state_dict(torch.load("./state/model_0.pt"))
    model.train()
    target = ConvDQNWithLength().cuda()
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    while global_rb.get_stored_size() < MIN_BUFFER:
        time.sleep(1)

    t = trange(num_episode)
    epsilon = max_epsilon
    for step in t:
        sample = global_rb.sample(BATCH_SIZE)

        unpacked_sample = unpack_sample(sample)
        q1, q2 = both_Q(model, target, unpacked_sample)
        loss = F.smooth_l1_loss(q1, q2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            q1 = Q_model(model, unpacked_sample[0], unpacked_sample[1], unpacked_sample[-2])
            absTD = torch.abs(q1 - q2).detach().cpu()
        global_rb.update_priorities(sample["indexes"], absTD)

        if step % TARGET_UPDATE_FREQ == 0:
            target.load_state_dict(model.state_dict())

        if step % EXPLORER_UPDATE_FREQ == 0:
            model_state = dict()
            for name, tensor in model.state_dict(keep_vars=True).items():
                model_state[name] = tensor.detach().clone().cpu()

            target_state = dict()
            for name, tensor in target.state_dict(keep_vars=True).items():
                target_state[name] = tensor.detach().clone().cpu()

            epsilon = compute_epsilon(step, min_epsilon, max_epsilon, epsilon_decay)
            for q in qs:
                q.put((model_state, target_state, epsilon))
        t.set_postfix(epsilon="%.3f" % epsilon, loss="%.3f" % loss.item())
    is_training_done.set()

    for p in ps:
        p.join()

    torch.save(target.state_dict(), "state/ConvDQNWithLength/model_1.pt")
