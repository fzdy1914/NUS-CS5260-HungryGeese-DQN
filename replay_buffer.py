from cpprb import ReplayBuffer

from parameters import *

env_dict = {
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

buffer = ReplayBuffer(BUFFER_SIZE, env_dict=env_dict)
