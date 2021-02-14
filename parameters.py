from kaggle_environments.envs.hungry_geese.hungry_geese import Action

ROW = 7
COLUMN = 11

ROW_CENTER = int((ROW - 1) / 2)
COLUMN_CENTER = int((COLUMN - 1) / 2)

HUNGER_RATE = 40
DEFAULT_NORMAL_REWARD = -1 / HUNGER_RATE
DEFAULT_FOOD_REWARD = 1
DEFAULT_HIT_REWARD = -5

NUM_ACTION = len(Action)
BUFFER_SIZE = 16384
LOCAL_BUFFER_SIZE = 256

MIN_BUFFER = 512  # minimum buffer to start training
BATCH_SIZE = 128

GAMMA = 0.99

LEARNING_RATE = 0.001

N_EXPLORER = 3

TARGET_UPDATE_FREQ = 50
EXPLORER_UPDATE_FREQ = 100


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

ACTION2NUM = {
    "EAST": 0,
    "SOUTH": 1,
    "WEST": 2,
    "NORTH": 3,
}

NUM2ACTION = {
    0: "EAST",
    1: "SOUTH",
    2: "WEST",
    3: "NORTH",
}

ENV_DICT = {
    "board": {
        "shape": (ROW, COLUMN)
    },
    "action": {},
    "reward": {},
    "next_board": {
        "shape": (ROW, COLUMN)
    },
    "done": {}
}

ENV_DICT_WITH_LENGTH = {
    "board": {
        "shape": (ROW, COLUMN)
    },
    "action": {},
    "reward": {},
    "next_board": {
        "shape": (ROW, COLUMN)
    },
    "done": {},
    "length": {},
    "next_length": {}
}

ENV_DICT_STACK = {
    "board": {
        "shape": (5, ROW, COLUMN)
    },
    "action": {},
    "reward": {},
    "next_board": {
        "shape": (5, ROW, COLUMN)
    },
    "done": {}
}
