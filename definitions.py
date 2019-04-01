import os

ROOT_DIR = os.path.dirname(__file__)
OBSTACLE_TOWER_DIR = os.path.join(ROOT_DIR, 'ObstacleTower')

# forward/backward/no-move
ACTION_MOVE = [0, 1, 2]

# left/right/no-move
ACTION_STRAFE = [0, 1, 2]

# clock/counterclock
ACTION_TURN = [0, 1, 2]

# no-op/jump
ACTION_JUMP = [0, 1]
