from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

from board import encode_observation
from parameters import NUM2ACTION

import torch

from model import ConvDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx = 0
prev_action = "NORTH"

model = ConvDQN()
model.load_state_dict(torch.load("model.pt"))
model.eval()


def agent(obs_dict, config_dict):
    global idx, prev_action

    board = torch.from_numpy(encode_observation(obs_dict, action=prev_action)).float()
    action = model.greedy(board.unsqueeze(0))
    action = NUM2ACTION[action.squeeze().item()]

    prev_action = action
    return action
