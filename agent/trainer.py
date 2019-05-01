import os
import torch
import numpy as np

from tqdm import tqdm
from tensorboardX import SummaryWriter

from agent.definitions import MODEL_PATH


class Trainer:
    def __init__(
        self,
        parallel_environment,
        experience,
        agent_network,
        action_space,
        num_envs,
        experience_history_size,
        batch_size,
        sequence_length,
        num_of_epoches,
        total_timesteps,
        learning_rate,
        device,
        ppo=False,
    ):

        self.env = parallel_environment
        self.experience = experience
        self.agent_network = agent_network
        self.action_space = action_space
        self.num_envs = num_envs
        self.experience_history_size = experience_history_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_of_epoches = num_of_epoches
        self.total_timesteps = total_timesteps
        self.distribution = torch.distributions.Categorical
        self.lr = learning_rate
        self.writer = SummaryWriter()
        self.optim = torch.optim.Adam(
            self.agent_network.parameters(), lr=self.lr, eps=1e-8
        )
        self.device = device
        self.ppo = ppo

    def sample_action(self, actions):
        """
        Sample action from tensor of action probabilities or log probabilities.
        Input: torch.Tensor([8, 54])
        Return: torch.Tensor([8])
        """
        return self.distribution(logits=actions).sample()

    def train(self):
        num_of_updates = self.total_timesteps // self.batch_size
        action_size = len(self.action_space)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=lambda step: (num_of_updates - step) / num_of_updates
        )

        for timestep in range(num_of_updates):
            pi_loss, val_loss, ent, pc_loss, v_loss, reward = self._fill_experience(
                action_size)
            self.writer.add_scalar("tower/rewards", reward, timestep)
            self.writer.add_scalar("tower/policy_loss", pi_loss, timestep)
            self.writer.add_scalar("tower/value_loss", val_loss, timestep)
            self.writer.add_scalar("tower/entropy_loss", ent, timestep)
            if pc_loss:
                self.writer.add_scalar("tower/pc_loss", pc_loss, timestep)
            if v_loss:
                self.writer.add_scalar("tower/vr_loss", v_loss, timestep)
            self.writer.add_scalar("tower/lr", np.array(lr_scheduler.get_lr()), timestep)

            lr_scheduler.step()
            if timestep % 100 == 0:
                name = "ppo" if self.ppo else "a2c"
                path = os.path.join(
                    MODEL_PATH, "model_{}_{}.bin".format(name, timestep)
                )
                torch.save(self.agent_network.state_dict(), path)

        self.writer.close()

    def _fill_experience(self, action_size):
        reset = True

        states = []
        reward_actions = []
        action_indices = []
        policy = []
        rewards = []
        values = []
        rhx = []
        chx = []
        dones = []

        counter = 0
        reward_acc = 0
        for _ in tqdm(range(self.batch_size)):
            with torch.no_grad():
                if reset:
                    old_state, key, old_time = self.env.reset()
                    last_rhs = None
                    reward_action = torch.zeros((self.num_envs, action_size + 1)).to(
                        self.device
                    )
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs
                    )
                    reset = False
                else:
                    last_rhs = self.experience.last_hidden_state
                    old_state, reward_action, old_time = self.experience.last_frames()
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs
                    )

                action = self.sample_action(policy_acts)
                new_actions = [self.action_space[act] for act in action]
                policies = policy_acts.view(action_size, self.num_envs)

                self.experience.last_hidden_state = rhs
                new_state, key, new_time, reward, done = self.env.step(new_actions)

                reward_acc += reward.mean()
                action_encoding = torch.zeros((action_size, self.num_envs))
                for i in range(self.num_envs):
                    action_encoding[action[i], i] = 1

                new_reward_action = self.experience.concatenate_reward_and_action(
                    reward, action_encoding)

                if len(torch.nonzero(done)) > 0:
                    self.env.reset()
                    reset = True
                    if not len(states):
                        continue
                    else:
                        break

                if len(torch.nonzero(reward)) > 0:
                    counter += 1

                states.append(new_state)
                reward_actions.append(new_reward_action)
                action_indices.append(action_encoding)
                policy.append(policies)
                rewards.append(self.experience.calculate_reward(
                    reward, new_time, old_time, key))
                values.append(value)
                rhx.append(rhs[0])
                chx.append(rhs[1])
                dones.append(done)

                self.experience.add_experience(
                    new_state,
                    old_state,
                    new_time,
                    old_time,
                    key,
                    reward,
                    action_encoding,
                    done,
                    value,
                    rhs[0],
                    rhs[1],
                    policies,
                )

        state_tensor = torch.cat(states, dim=0)
        reward_actions = torch.cat(reward_actions, dim=1).transpose_(1, 0)
        action_indices = torch.cat(action_indices, dim=1).transpose_(1, 0)
        policy_tensor = torch.cat(policy, dim=0)
        hidden_state = (torch.cat(rhx, dim=1), torch.cat(chx, dim=1))

        values = torch.stack(values)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        pc_loss = None
        v_loss = None
        if self.experience.full():
            with torch.no_grad():
                _, q_aux = self.agent_network.pixel_control_act(
                    new_state, new_reward_action.transpose_(1, 0), rhs)
            pc_loss = self.pc_control(action_size, q_aux)
            v_loss = self.value_replay(action_size)
            agent_loss, pi_loss, value_loss, ent = self.base_loss(action_size,
                                                                  state_tensor,
                                                                  reward_actions,
                                                                  action_indices,
                                                                  rewards,
                                                                  values,
                                                                  policy_tensor,
                                                                  hidden_state,
                                                                  dones)
            loss = agent_loss + v_loss + pc_loss
        else:
            agent_loss, pi_loss, value_loss, ent = self.base_loss(action_size,
                                                                  state_tensor,
                                                                  reward_actions,
                                                                  action_indices,
                                                                  rewards,
                                                                  values,
                                                                  policy_tensor,
                                                                  hidden_state,
                                                                  dones)
            loss = agent_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        print("Hits in episode run: {}".format(counter))
        return pi_loss, value_loss, ent, pc_loss, v_loss, reward_acc

    def base_loss(
        self,
        action_size,
        states,
        reward_actions,
        action_indices,
        rewards,
        values,
        old_policy,
        base_rhs,
        dones,
    ):
        batch_returns = []
        batch_advantages = []

        for env in range(self.num_envs):
            returns = self.experience.compute_returns(
                rewards[:, env], values[:, env], dones[:, env]
            )
            returns = torch.Tensor(returns).to(self.device)

            adv = returns - values[:, env]

            batch_advantages.append(adv)
            batch_returns.append(returns)

        advantage = torch.cat(batch_advantages, dim=0)
        if self.ppo:
            advantage = (advantage - torch.mean(advantage, dim=0)) / torch.std(
                advantage + 1e-6
            )

        new_value, policy_acts, _ = self.agent_network.act(
            states, reward_actions, base_rhs
        )
        returns = torch.cat(batch_returns, dim=0).to(self.device)

        if self.ppo:
            agent_loss, pi_loss, v_loss, entropy = self.agent_network.ppo_loss(
                old_policy, policy_acts, advantage, returns, new_value, action_indices
            )
        else:
            agent_loss, pi_loss, v_loss, entropy = self.agent_network.a2c_loss(
                policy_acts, advantage, returns, new_value, action_indices
            )

        return agent_loss, pi_loss, v_loss, entropy

    def pc_control(self, action_size, q_aux):
        exp_batches = self.experience.sample_observations(self.sequence_length)
        states, reward_actions, action_indices, _, rewards, _, pixel_controls, rhs, dones = (
            exp_batches
        )

        batch_pc_returns = []

        for env in range(self.num_envs):
            pc_returns = self.experience.compute_pc_returns(
                q_aux[env, :, :], rewards[env], pixel_controls[env], dones[env]
            )
            batch_pc_returns.append(pc_returns)

        pc_returns = torch.cat(batch_pc_returns, dim=0).to(self.device)

        q_aux, _ = self.agent_network.pixel_control_act(states, reward_actions, rhs)

        pc_loss = self.agent_network.pc_loss(
            action_size, action_indices, q_aux, pc_returns
        )

        return pc_loss

    def value_replay(self, action_size):
        exp_batches = self.experience.sample_observations(self.sequence_length)
        states, reward_actions, _, _, rewards, values, _, rhs, dones = exp_batches

        batch_v_returns = []

        for env in range(self.num_envs):
            v_returns = self.experience.compute_v_returns(
                rewards[env], values[env], dones[env]
            )
            v_returns = torch.Tensor(v_returns).to(self.device)
            batch_v_returns.append(v_returns)

        new_value, _, _ = self.agent_network.act(states, reward_actions, rhs)

        v_returns = torch.cat(batch_v_returns, dim=0).to(self.device)
        v_loss = self.agent_network.v_loss(v_returns, new_value)

        return v_loss
