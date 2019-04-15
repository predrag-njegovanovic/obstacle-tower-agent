import agent.definitions as definitions

from obstacle_tower_env import ObstacleTowerEnv
from agent.tower_agent import TowerAgent
from agent.utils import create_action_space


if __name__ == "__main__":
    env_path = definitions.OBSTACLE_TOWER_PATH
    model_path = definitions.MODEL_PATH

    env = ObstacleTowerEnv(env_path, realtime_mode=True)
    # resize input frame

    config = definitions.network_params

    actions = create_action_space()
    action_size = len(actions)
