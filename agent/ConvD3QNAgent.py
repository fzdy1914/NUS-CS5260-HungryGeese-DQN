import sys
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate, Observation, adjacent_positions
import torch

sys.path.append('/kaggle_simulations/agent/')
from parameters import ROW, COLUMN, NUM2ACTION
from model import ConvD3QN_4
from board import encode_observation


prev_action = "NORTH"

model = ConvD3QN_4()
model.load_state_dict(torch.load('/kaggle_simulations/agent/model.pt', map_location=torch.device('cpu')))
model.eval()


def agent(obs_dict, config_dict):
    global prev_action

    observation = Observation(obs_dict)

    board = torch.from_numpy(encode_observation(obs_dict, action=prev_action)).float()
    action_list = model.forward(board.unsqueeze(0)).squeeze().topk(k=4)
    action = NUM2ACTION[action_list[1][0].item()]
    if Action[prev_action].opposite() == Action[action]:
        print("0. try opposite")
        print("0. prev:", action, "0.now:", NUM2ACTION[action_list[1][1].item()])
        action = NUM2ACTION[action_list[1][1].item()]

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

    print("other_body", other_goose_body)
    if len(self_goose) > 0:
        self_goose_head = self_goose[0]
        next_pos_idx = translate(self_goose_head, Action[action], COLUMN, ROW)
        if next_pos_idx in other_goose_body:
            print("try hit")
            for act in range(1, 4):
                temp_action = NUM2ACTION[action_list[1][act].item()]
                if Action[prev_action].opposite() == Action[temp_action]:
                    continue
                if translate(self_goose_head, Action[temp_action], COLUMN, ROW) not in other_goose_body:
                    print("1. prev:", action, "1.now:", temp_action)
                    action = temp_action
                    break

        if next_pos_idx in food_list:
            for idx in adjacent_positions(next_pos_idx, COLUMN, ROW):
                finish = False
                if idx in other_goose_head:
                    print("danger pos")
                    for act in range(1, 4):
                        temp_action = NUM2ACTION[action_list[1][act].item()]
                        if Action[prev_action].opposite() == Action[temp_action]:
                            continue
                        if translate(self_goose_head, Action[temp_action], COLUMN, ROW) not in other_goose_body:
                            print("2. prev:", action, "2.now:", temp_action)
                            action = temp_action
                            finish = True
                            break

                if finish:
                    break

    prev_action = action
    return action
