from kaggle_environments.envs.hungry_geese.hungry_geese import Action

ROW = 7
COLUMN = 11

WRAPPED_ROW = ROW + 2
WRAPPED_COLUMN = COLUMN + 2

HUNGER_RATE = 40
N_REWARD = -1 / HUNGER_RATE

NUM_ACTION = len(Action)
BUFFER_SIZE = 8196
BATCH_SIZE = 128


class Grid:
    FOOD = -1
    EMPTY = 0
    GOOSE_HEAD = 1
    GOOSE_BODY = 2
    GOOSE_TAIL = 3

    OTHER_HEAD = 4
    OTHER_TAIL = 5


STATUS = {
    "ACTIVE": 0,
    "DONE": 1
}

action2num = {
    "EAST": 0,
    "SOUTH": 1,
    "WEST": 2,
    "NORTH": 3,
}

num2action = {
    0: "EAST",
    1: "SOUTH",
    2: "WEST",
    3: "NORTH",
}