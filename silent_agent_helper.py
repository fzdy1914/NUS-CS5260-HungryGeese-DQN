from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate

from parameters import COLUMN, ROW


def can_rotate(goose):
    orders = [
        [Action.WEST, Action.NORTH, Action.EAST],
        [Action.SOUTH, Action.WEST, Action.NORTH],
        [Action.EAST, Action.SOUTH, Action.WEST],
        [Action.NORTH, Action.EAST, Action.SOUTH],

        [Action.EAST, Action.NORTH, Action.WEST],
        [Action.SOUTH, Action.EAST, Action.NORTH],
        [Action.WEST, Action.SOUTH, Action.EAST],
        [Action.NORTH, Action.WEST, Action.SOUTH],
    ]
    for order in orders:
        pos = goose[0]

        done = True
        for i in range(3):
            pos = translate(pos, order[i], COLUMN, ROW)
            if pos != goose[i + 1]:
                done = False
                break
        if done:
            return True, order[1].name
    return False, None


def get_available_action(pos, other_goose_body, prev_action):
    action_list = list()
    for action in Action:
        if translate(pos, action, COLUMN, ROW) not in other_goose_body and action.opposite() != Action[prev_action]:
            action_list.append(action.name)
    return action_list


def get_adjacent_action(action):
    if action in ["NORTH", "SOUTH"]:
        return ["EAST", "WEST"]
    else:
        return ["NORTH", "SOUTH"]


def get_active_geese_num(goose):
    num = 0
    for geese in goose:
        if len(geese) > 0:
            num += 1
    return num