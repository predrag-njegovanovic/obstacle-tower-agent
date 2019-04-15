import os
import multiprocessing

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
OBSTACLE_TOWER_DIR = os.path.join(ROOT_DIR, "ObstacleTower")
OBSTACLE_TOWER_PATH = os.path.join(OBSTACLE_TOWER_DIR, "obstacletower")

MODEL_PATH = os.path.join(ROOT_DIR, "models")

# forward/backward/no-move
ACTION_MOVE = [0, 1, 2]

# left/right/no-move
ACTION_STRAFE = [0, 1, 2]

# clock/counterclock
ACTION_TURN = [0, 1, 2]

# no-op/jump
ACTION_JUMP = [0, 1]

# Training parameters
NUM_ENVS = multiprocessing.cpu_count()
EPOCHES = 32
BATCH_SIZE = 20
TIMESTAMPS = 10000000
OBSERVATION_SIZE = 2000

network_params = {
    "first_layer": 16,
    "second_layer": 32,
    "conv_output": 256,
    "hidden_state_size": 256,
}
