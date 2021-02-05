from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

idx = 0


def agent(obs_dict, config_dict):
    global idx
    dic = {
        0: Action.NORTH.name,
        1: Action.EAST.name,
        2: Action.SOUTH.name,
        3: Action.WEST.name,
    }
    idx = (idx + 1) % 4
    return dic[idx]
