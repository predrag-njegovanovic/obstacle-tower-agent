import torch
import argparse
import multiprocessing

import agent.definitions as definitions

from agent.trainer import Trainer
from agent.tower_agent import TowerAgent
from agent.experience_memory import ExperienceMemory
from agent.parallel_environment import ParallelEnvironment
from agent.utils import create_action_space, mean_std_obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Obstacle Tower Agent")

    parser.add_argument(
        "--num_envs",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel environment to train on.",
    )
    parser.add_argument(
        "--experience_memory", type=int, default=128, help="Size of experience memory."
    )
    parser.add_argument(
        "--timesteps", type=int, default=5000000, help="Number of training steps."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Number of steps per update"
    )
    parser.add_argument(
        "--epoches",
        type=int,
        default=4,
        help="Number of updates once the experience memory is filled.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=4e-4, help="Learning rate."
    )
    parser.add_argument(
        "--ppo", type=bool, default=False, help="Use PPO algorithm for training."
    )
    parser.add_argument("--use_cuda", type=bool, default=True, help="Use GPU training.")

    args = parser.parse_args()
    config = definitions.network_params

    actions = create_action_space()
    action_size = len(actions)

    env_path = definitions.OBSTACLE_TOWER_PATH
    env = ParallelEnvironment(env_path, args.num_envs)
    env.start_parallel_execution()
    obs_mean, obs_std = mean_std_obs(10000)

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
    agent.to_cuda()
    if args.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    memory = ExperienceMemory(
        args.num_envs, args.experience_memory, action_size, device
    )

    trainer = Trainer(
        env,
        memory,
        agent,
        actions,
        args.num_envs,
        args.experience_memory,
        args.batch_size,
        args.epoches,
        args.timesteps,
        args.learning_rate,
        device,
        args.ppo,
    )
    trainer.train()
