import os
import torch
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

from agent.definitions import MODEL_PATH, UPDATE_CYCLES


class RewardForwardFilter:
    def __init__(self, batch_size, num_envs, gamma):
        self.running_reward = torch.zeros(batch_size, num_envs)
        self.gamma = gamma

    def update(self, step, reward):
        self.running_reward[step, :] = (
            self.running_reward[step, :] * self.gamma + reward
        )


class Trainer:
    def __init__(
        self,
        parallel_environment,
        experience,
        agent_network,
        action_space,
        num_envs,
        experience_size,
        batch_size,
        num_of_epochs,
        total_timesteps,
        learning_rate,
        device,
        ppo=False,
    ):

        self.env = parallel_environment
        self.experience = experience
        self.agent_network = agent_network
        self.action_space = action_space
        self.action_size = len(action_space)
        self.num_envs = num_envs
        self.experience_size = experience_size
        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs
        self.total_timesteps = total_timesteps
        self.distribution = torch.distributions.Categorical
        self.lr = learning_rate
        self.writer = SummaryWriter()
        self.optim = torch.optim.Adam(
            self.agent_network.parameters(), lr=self.lr, eps=1e-5
        )
        self.device = device
        self.ppo = ppo
        self.reward_updater = RewardForwardFilter(batch_size, num_envs, 0.9)

    def sample_action(self, actions):
        """
        Sample action from tensor of action probabilities or log probabilities.
        Input: torch.Tensor([8, 54])
        Return: torch.Tensor([8])
        """
        return self.distribution(probs=actions).sample()

    def train(self):
        num_of_updates = self.total_timesteps // (self.num_envs * self.batch_size)

        learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=lambda step: 1 - (step / float(num_of_updates))
        )

        for timestep in range(num_of_updates):
            episode_reward = self.collect_experience()
            policy_loss, value_loss, entropy_loss, forward_loss, inverse_loss = (
                self.agent_update()
            )
            self.writer.add_scalars(
                "tower/rewards",
                {
                    "mean": torch.mean(episode_reward),
                    "std": torch.std(episode_reward),
                    "max": torch.max(episode_reward),
                },
                timestep,
            )
            self.writer.add_scalar("tower/policy_loss", policy_loss.mean(), timestep)
            self.writer.add_scalar("tower/value_loss", value_loss.mean(), timestep)
            self.writer.add_scalar("tower/entropy_loss", entropy_loss.mean(), timestep)
            self.writer.add_scalar("tower/forward_loss", forward_loss.mean(), timestep)
            self.writer.add_scalar("tower/inverse_loss", inverse_loss.mean(), timestep)
            self.writer.add_scalar(
                "tower/lr", np.array(learning_rate_scheduler.get_lr()), timestep
            )

            learning_rate_scheduler.step()
            self.experience.empty()
            if timestep % 250 == 0:
                name = "ppo" if self.ppo else "a2c"
                path = os.path.join(
                    MODEL_PATH, "model_{}_{}.bin".format(name, timestep)
                )
                torch.save(self.agent_network.state_dict(), path)

        self.writer.close()

    def collect_experience(self):
        reset = True
        counter = 0
        episode_reward = torch.zeros(self.num_envs)

        for episode_step in tqdm(range(self.batch_size)):
            with torch.no_grad():
                if reset:
                    state, key, time = self.env.reset()
                    last_rhs = torch.zeros((1, self.num_envs, 512)).to(self.device)
                    value, policy, rhs = self.agent_network.act(state, last_rhs)
                    reset = False
                else:
                    last_rhs = self.experience.last_hidden_state
                    state = self.experience.last_states()
                    value, policy, rhs = self.agent_network.act(state, last_rhs)

                action = self.sample_action(policy)
                new_actions = [self.action_space[act] for act in action]

                self.experience.last_hidden_state = rhs
                new_state, key, new_time, reward, done = self.env.step(new_actions)

                if len(torch.nonzero(done)):
                    break

                if len(torch.nonzero(reward)):
                    counter += 1

                action_encoding = torch.zeros((self.num_envs, self.action_size)).to(
                    self.device
                )

                for i in range(self.num_envs):
                    action_encoding[i, action[i]] = 1

                intrinsic_reward, state_features, new_state_features = self.agent_network.icm_act(
                    state, new_state, action_encoding
                )

                total_reward = reward + intrinsic_reward.cpu()
                episode_reward += total_reward

                self.reward_updater.update(episode_step, total_reward)
                self.experience.add_experience(
                    new_state,
                    state,
                    total_reward,
                    action_encoding,
                    done,
                    value,
                    policy,
                    state_features,
                    new_state_features,
                )
                self.experience.increase_frame_pointer()

        print("Hits in episode run: {}".format(counter))
        return episode_reward

    def agent_update(self):
        total_episode_steps = UPDATE_CYCLES * self.num_of_epochs

        value_loss = torch.zeros(total_episode_steps)
        policy_loss = torch.zeros(total_episode_steps)
        entropy_loss = torch.zeros(total_episode_steps)
        forward_loss = torch.zeros(total_episode_steps)
        inverse_loss = torch.zeros(total_episode_steps)

        memory_pointer = self.experience.memory_pointer
        running_reward = self.reward_updater.running_reward[: memory_pointer - 1, :]
        running_reward_std = torch.std(running_reward, dim=0)

        for update in range(UPDATE_CYCLES):
            for epoch in tqdm(range(self.num_of_epochs)):
                if self.ppo:
                    minibatch_size = self.num_envs // self.num_of_epochs
                    experience_batches = self.experience.ppo_policy_sampling(
                        minibatch_size, running_reward_std
                    )
                else:
                    experience_batches = self.experience.a2c_policy_sampling(
                        running_reward_std
                    )

                if self.ppo:
                    agent_loss, policy, value, entropy, forward, inverse = self.ppo_loss(
                        minibatch_size, *experience_batches
                    )
                else:
                    agent_loss, policy, value, entropy, forward, inverse = self.a2c_loss(
                        *experience_batches
                    )

                self.optim.zero_grad()
                loss = agent_loss / self.num_of_epochs
                loss.backward()

                value_loss[update * self.num_of_epochs + epoch].copy_(value)
                policy_loss[update * self.num_of_epochs + epoch].copy_(policy)
                entropy_loss[update * self.num_of_epochs + epoch].copy_(entropy)
                forward_loss[update * self.num_of_epochs + epoch].copy_(forward)
                inverse_loss[update * self.num_of_epochs + epoch].copy_(inverse)

            torch.nn.utils.clip_grad_norm_(self.agent_network.parameters(), 40)
            self.optim.step()

        return (policy_loss, value_loss, entropy_loss, forward_loss, inverse_loss)

    def a2c_loss(
        self,
        states,
        action_indices,
        rewards,
        values,
        dones,
        state_features,
        new_state_features,
    ):

        returns = self.experience.compute_returns(rewards, values, dones)
        returns = torch.Tensor(returns).to(self.device)

        advantage = returns - values
        advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-6)

        rhs = torch.zeros((1, 1, 512)).to(self.device)
        new_value, policy_acts, _ = self.agent_network.act(states, rhs)

        predicted_states = self.agent_network.forward_act(
            state_features, action_indices
        )
        predicted_acts = self.agent_network.inverse_act(
            state_features, new_state_features
        )

        losses = self.agent_network.a2c_loss(
            policy_acts,
            advantage,
            returns,
            new_value,
            action_indices,
            new_state_features,
            predicted_states,
            predicted_acts,
        )

        return losses

    def ppo_loss(
        self,
        minibatch_size,
        states,
        action_indices,
        old_policy,
        rewards,
        values,
        dones,
        state_features,
        new_state_features,
    ):
        batch_returns = []
        batch_advantages = []

        for env in range(minibatch_size):
            returns = self.experience.compute_returns(
                rewards[env], values[env], dones[env]
            )

            returns = torch.Tensor(returns).to(self.device)

            advantage = returns - values[env]

            batch_advantages.append(advantage)
            batch_returns.append(returns)

        returns = torch.cat(batch_returns, dim=0).to(self.device)
        advantage = torch.cat(batch_advantages, dim=0)
        advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-6)

        rhs = torch.zeros((1, 1, 512)).to(self.device)
        new_value, policy_acts, _ = self.agent_network.act(states, rhs)

        predicted_states = self.agent_network.forward_act(
            state_features, action_indices
        )
        predicted_acts = self.agent_network.inverse_act(
            state_features, new_state_features
        )

        losses = self.agent_network.ppo_loss(
            old_policy,
            policy_acts,
            advantage,
            returns,
            new_value,
            action_indices,
            new_state_features,
            predicted_states,
            predicted_acts,
        )
        return losses
