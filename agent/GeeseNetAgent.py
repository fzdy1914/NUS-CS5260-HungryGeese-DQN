import sys

import numpy as np
import torch

sys.path.append('/kaggle_simulations/agent/')
from geese_net import GeeseNet, make_input


model = GeeseNet()
model.load_state_dict(torch.load('/kaggle_simulations/agent/model.pt', map_location=torch.device('cpu')))
model.eval()

# Main Function of Agent
obses = []


def agent(obs, _):
    obses.append(obs)
    x = make_input(obses)
    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)
        o = model(xt)
    p = o['policy'].squeeze(0).detach().numpy()

    actions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    return actions[np.argmax(p)]
