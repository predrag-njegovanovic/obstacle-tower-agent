import torch
import argparse
import multiprocessing

import agent.definitions as definitions

from agent.trainer import Trainer
from agent.tower_agent import TowerAgent
from agent.experience_memory import ExperienceMemory
from agent.parallel_environment import ParallelEnvironment
from agent.utils import create_action_space, log_uniform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Obstacle Tower Agent")

    parser.add_argument(
        "--num_envs",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel environment to train on.",
    )
    parser.add_argument(
        "--observation_size", type=int, default=2000, help="Size of experience memory."
    )
    parser.add_argument(
        "--lr_low_rate",
        type=float,
        default=1e-4,
        help="Learning rate is sampled from log uniform distribution(low, high).",
    )
    parser.add_argument(
        "--lr_high_rate",
        type=float,
        default=5e-4,
        help="Learning rate is sampled from log uniform distribution(low, high).",
    )
    parser.add_argument(
        "--entropy_low_rate",
        type=float,
        default=1e-5,
        help="Entropy coefficient is sampled from log uniform distribution(low, high).",
    )
    parser.add_argument(
        "--entropy_high_rate",
        type=float,
        default=1e-5,
        help="Entropy coefficient is sampled from log uniform distribution(low, high).",
    )
    parser.add_argument(
        "--pc_low_rate",
        type=float,
        default=0.01,
        help="""Pixel control lambda coefficient is sampled
                from log uniform distribution(low, high).""",
    )
    parser.add_argument(
        "--pc_high_rate",
        type=float,
        default=0.1,
        help="""Pixel control lambda coefficient is sampled
                from log uniform distribution(low, high).""",
    )
    parser.add_argument(
        "--timesteps", type=int, default=2500000, help="Number of training steps."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Numer of samples from experience memory.",
    )
    parser.add_argument(
        "--epoches",
        type=int,
        default=64,
        help="Number of updates once the experience memory is filled.",
    )
    parser.add_argument("--use_cuda", type=bool, default=True, help="Use GPU training.")

    args = parser.parse_args()
    config = definitions.network_params

    actions = create_action_space()
    action_size = len(actions)

    env_path = definitions.OBSTACLE_TOWER_PATH
    env = ParallelEnvironment(env_path, args.num_envs)
    env.start_parallel_execution()

    learning_rate = log_uniform(args.lr_low_rate, args.lr_high_rate)
    entropy_coeff = log_uniform(args.entropy_low_rate, args.entropy_high_rate)
    pc_lambda = log_uniform(args.pc_low_rate, args.pc_high_rate)

    agent = TowerAgent(
        action_size,
        config["first_filters"],
        config["second_filters"],
        config["convolution_output"],
        config["hidden_state"],
        entropy_coeff=entropy_coeff,
        pc_lambda=pc_lambda,
    )
    agent.to_cuda()
    if args.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    memory = ExperienceMemory(args.num_envs, args.observation_size, action_size, device)

    trainer = Trainer(
        env,
        memory,
        agent,
        actions,
        args.num_envs,
        args.observation_size,
        args.batch_size,
        args.epoches,
        args.timesteps,
        learning_rate,
        device,
    )
    trainer.train()
