import os
import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter

from agent.definitions import MODEL_PATH


class Trainer:
    def __init__(
        self,
        parallel_environment,
        experience_memory,
        agent_network,
        action_space,
        num_envs,
        experience_history_size,
        batch_size,
        num_of_epoches,
        total_timesteps,
        learning_rate,
        device,
    ):

        self.env = parallel_environment
        self.experience_memory = experience_memory
        self.agent_network = agent_network
        self.action_space = action_space
        self.num_envs = num_envs
        self.experience_history_size = experience_history_size
        self.batch_size = batch_size
        self.num_of_epoches = num_of_epoches
        self.total_timesteps = total_timesteps
        self.distribution = torch.distributions.Categorical
        self.lr = learning_rate
        self.writer = SummaryWriter()
        self.optim = torch.optim.Adam(
            self.agent_network.parameters(), lr=self.lr, eps=1e-5
        )
        self.device = device

    def sample_action(self, actions):
        """
        Sample action from tensor of action probabilities or log probabilities.
        Input: torch.Tensor([8, 54])
        Return: torch.Tensor([8])
        """
        return self.distribution(actions).sample()

    def train(self):
        num_of_updates = self.total_timesteps // self.experience_history_size
        action_size = len(self.action_space)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=lambda step: (num_of_updates - step) / num_of_updates
        )

        for timestep in range(num_of_updates):
            self._fill_experience(timestep, action_size)
            pi_loss, v_loss, entropy, pc_loss, vr_loss = self._update_observations(
                action_size
            )
            mean_reward = self.experience_memory.mean_reward()

            self.writer.add_scalar("tower/mean_reward", mean_reward, timestep)
            self.writer.add_scalar("tower/policy_loss", torch.mean(pi_loss), timestep)
            self.writer.add_scalar("tower/value_loss", torch.mean(v_loss), timestep)
            self.writer.add_scalar("tower/entropy_loss", torch.mean(entropy), timestep)
            self.writer.add_scalar("tower/pc_loss", torch.mean(pc_loss), timestep)
            self.writer.add_scalar("tower/vr_loss", torch.mean(vr_loss), timestep)

            lr_scheduler.step()
            self.experience_memory.empty()
            if timestep % 100 == 0:
                path = os.path.join(MODEL_PATH, "model_{}.bin".format(timestep))
                torch.save(self.agent_network.state_dict(), path)

        self.writer.close()

    def _lr(self, lr_scheduler):
        return lr_scheduler.get_lr()

    def _fill_experience(self, timestep, action_size):
        for step in tqdm(range(self.experience_history_size)):
            with torch.no_grad():
                if not timestep:
                    old_state, key, old_time = self.env.reset()
                    reward_action = torch.zeros((self.num_envs, action_size + 1)).to(
                        self.device
                    )
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action
                    )
                    _, q_aux_max = self.agent_network.pixel_control_act(
                        old_state, reward_action
                    )
                else:
                    last_rhs = self.experience_memory.last_hidden_state
                    old_state, reward_action, old_time = (
                        self.experience_memory.last_frames()
                    )
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs
                    )
                    _, q_aux_max = self.agent_network.pixel_control_act(
                        old_state, reward_action, last_rhs
                    )

                action = self.sample_action(policy_acts)
                new_actions = [self.action_space[act] for act in action]

                self.experience_memory.last_hidden_state = rhs
                new_state, key, new_time, reward, done = self.env.step(new_actions)

                action_encoding = torch.zeros((action_size, self.num_envs))
                policies = policy_acts.view(action_size, self.num_envs)
                for i in range(self.num_envs):
                    action_encoding[action[i], i] = 1

                self.experience_memory.add_experience(
                    new_state,
                    old_state,
                    new_time,
                    old_time,
                    key,
                    reward,
                    action_encoding,
                    done,
                    value,
                    policies,
                    q_aux_max,
                )
                self.experience_memory.increase_frame_pointer()
                timestep += 1

    def _update_observations(self, action_size):
        epoches = self.num_of_epoches

        pc_loss = torch.zeros(epoches)
        value_loss = torch.zeros(epoches)
        policy_loss = torch.zeros(epoches)
        entropy_loss = torch.zeros(epoches)
        value_replay_loss = torch.zeros(epoches)

        for i in tqdm(range(epoches)):
            exp_batches = self.experience_memory.sample_observations(self.batch_size)
            states, reward_actions, action_indices, rewards, values, q_auxes, pixel_controls = (
                exp_batches
            )

            batch_returns = []
            batch_v_returns = []
            batch_pc_returns = []
            batch_advantages = []

            for env in range(self.num_envs):
                returns = self.experience_memory.compute_returns(
                    rewards[env], values[env]
                )
                pc_returns = self.experience_memory.compute_pc_returns(
                    q_auxes[env], rewards[env], pixel_controls[env]
                )
                v_returns = self.experience_memory.compute_v_returns(
                    rewards[env], values[env]
                )
                returns = torch.Tensor(returns).to(self.device)
                v_returns = torch.Tensor(v_returns).to(self.device)

                adv = returns - values[env]

                batch_advantages.append(adv)
                batch_returns.append(returns)
                batch_v_returns.append(v_returns)
                batch_pc_returns.append(pc_returns)

            advantage = torch.cat(batch_advantages, dim=0).unsqueeze(-1)
            advantage = (advantage - torch.mean(advantage, dim=0)) / torch.std(
                advantage + 1e-6
            )

            new_value, policy_acts, _ = self.agent_network.act(states, reward_actions)
            q_aux, _ = self.agent_network.pixel_control_act(states, reward_actions)

            returns = torch.cat(batch_returns, dim=0).to(self.device)
            pc_returns = torch.cat(batch_pc_returns, dim=0).to(self.device)
            v_returns = torch.cat(batch_v_returns, dim=0).to(self.device)

            a2c_loss, pi_loss, v_loss, entropy = self.agent_network.a2c_loss(
                policy_acts, advantage, returns, new_value, action_indices
            )

            pixel_control_loss = self.agent_network.pc_loss(
                action_size, action_indices, q_aux, pc_returns
            )

            v_loss = self.agent_network.v_loss(v_returns, new_value)

            loss = a2c_loss + v_loss + pixel_control_loss

            value_loss[i].copy_(v_loss)
            policy_loss[i].copy_(pi_loss)
            entropy_loss[i].copy_(entropy)
            pc_loss[i].copy_(pixel_control_loss)
            value_replay_loss.copy_(v_loss)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # ppo loss
        return policy_loss, value_loss, entropy_loss, pc_loss, value_replay_loss
