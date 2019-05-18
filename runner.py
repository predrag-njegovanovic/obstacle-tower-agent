import os
import torch
import argparse

import agent.definitions as definitions

from obstacle_tower_env import ObstacleTowerEnv
from agent.tower_agent import TowerAgent
from agent.utils import create_action_space, mean_std_obs
from agent.parallel_environment import prepare_state


def greedy_policy(action_space, policy):
    print(policy)
    probs = torch.distributions.Categorical
    index = probs(probs=policy).sample()
    # index = torch.argmax(policy)
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
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=True,
        help="Use GPU for inference phase. This will transfer model and tensors to VRAM.",
    )

    args = parser.parse_args()

    env_path = definitions.OBSTACLE_TOWER_PATH
    model_name = os.path.join(definitions.MODEL_PATH, args.model_name)
    obs_mean, obs_std = mean_std_obs(10000)

    env = ObstacleTowerEnv(env_path, retro=False, realtime_mode=True, worker_id=10)
    env.seed(args.seed)
    env.floor(1)
    env.reset()

    config = definitions.network_params
    actions = create_action_space()
    action_size = len(actions)

    agent = TowerAgent(
        action_size,
        config["first_filters"],
        config["second_filters"],
        config["convolution_output"],
        config["hidden_state"],
        config["feature_ext_filters"],
        config["feature_output_size"],
        config["forward_model_f_layer"],
        config["inverse_model_f_layer"],
        obs_mean,
        obs_std,
    )

    agent.load_state_dict(torch.load(model_name))

    if args.use_cuda:
        agent.to_cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    frame, key, time = env.reset()
    state = torch.Tensor(prepare_state(frame)).unsqueeze(0).to(device)

    action_encoding = torch.zeros((action_size, 1)).to(device)
    reward_action = torch.zeros((1, action_size + 1)).to(device)

    # Bootstrap initial state and reward_action vector
    value, policy, rhs = agent.act(state, reward_action)
    action, action_index = greedy_policy(actions, policy)
    action_encoding[action_index] = 1
    while True:
        for _ in range(6):
            obs, reward, done, _ = env.step(action)
            frame, _, _ = obs

        state = torch.Tensor(prepare_state(frame)).unsqueeze(0).to(device)
        if done:
            break

        reward_tensor = torch.Tensor([reward]).unsqueeze(1).to(device)
        temporary_tensor = torch.cat((action_encoding, reward_tensor))
        reward_action.copy_(temporary_tensor.transpose_(0, 1))

        value, policy, rhs = agent.act(state, reward_action, rhs)
        action, action_index = greedy_policy(actions, policy)

        action_encoding = torch.zeros((action_size, 1)).cuda()
        action_encoding[action_index] = 1
