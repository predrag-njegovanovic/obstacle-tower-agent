import os
import torch
import argparse

import agent.definitions as definitions

from obstacle_tower_env import ObstacleTowerEnv
from agent.tower_agent import TowerAgent
from agent.utils import create_action_space
from agent.parallel_environment import prepare_state


def greedy_policy(action_space, policy):
    value, index = torch.max(policy, 0)
    return action_space[index], index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obstacle Tower Agent")
    parser.add_argument(
        "--model_name",
        type=str,
        default="model_0.bin",
        help="Name of model to use. E.g. model_(num_of_update).bin",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment can use seed. Default seed is 0.",
    )

    args = parser.parse_args()

    env_path = definitions.OBSTACLE_TOWER_PATH
    model_name = os.path.join(definitions.MODEL_PATH, args.model_name)

    env = ObstacleTowerEnv(env_path, realtime_mode=True)
    env.seed(args.seed)

    config = definitions.network_params
    actions = create_action_space()
    action_size = len(actions)

    agent = TowerAgent(
        action_size,
        config["first_filters"],
        config["second_filters"],
        config["convolution_output"],
        config["hidden_state"],
    )

    agent.load_state_dict(model_name)
    agent.to_cuda()

    frame, _, _ = env.reset()
    state = prepare_state(frame)
    action_encoding = torch.zeros((action_size, 1)).cuda()
    reward_action = torch.zeros((1, action_size + 1)).cuda()

    value, policy, rhs = agent.act(state, reward_action)
    action, action_index = greedy_policy(actions, policy)
    action_encoding[action_index] = 1
    while True:
        frame, _, _, reward, done = env.step(action)
        state = prepare_state(frame)
        if done:
            break

        reward_action.copy_(torch.cat((action_encoding, reward), dim=0))
        value, policy, rhs = agent.act(state, reward_action, rhs)
        action, action_index = greedy_policy(actions, policy)
