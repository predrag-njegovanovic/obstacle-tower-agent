import torch
import numpy as np
import itertools

from agent import definitions


def log_uniform(low, high, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


def create_action_space():
    actions = itertools.product(
        definitions.ACTION_MOVE,
        definitions.ACTION_STRAFE,
        definitions.ACTION_TURN,
        definitions.ACTION_JUMP,
    )
    action_space = [list(action) for action in actions]
    return action_space


def torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
