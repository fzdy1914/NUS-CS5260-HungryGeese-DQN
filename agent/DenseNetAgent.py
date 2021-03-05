import sys
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate, Observation, adjacent_positions
import torch

sys.path.append('/kaggle_simulations/agent/')
from parameters import ROW, COLUMN, NUM2ACTION
from dense_model import DenseNet
from board import encode_observation
from silent_agent_helper import get_available_action

prev_action = "NORTH"

model = DenseNet()
model.load_state_dict(torch.load('/kaggle_simulations/agent/model.pt', map_location=torch.device('cpu')))
model.eval()


epoch = 0


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

    board = torch.from_numpy(encode_observation(obs_dict, action=prev_action)).float()
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
