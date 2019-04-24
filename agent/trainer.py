import os
import torch

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
            self.agent_network.parameters(), lr=self.lr, eps=1e-5
        )
        self.device = device
        self.ppo = ppo

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
            pi_loss, v_loss, entropy, pc_loss, vr_loss, door_hit = self._update_observations(
                action_size
            )
            mean_reward = self.experience.reward_stats()

            self.writer.add_scalar("tower/rewards", mean_reward, timestep)
            self.writer.add_scalar("tower/policy_loss", torch.mean(pi_loss), timestep)
            self.writer.add_scalar("tower/value_loss", torch.mean(v_loss), timestep)
            self.writer.add_scalar("tower/entropy_loss", torch.mean(entropy), timestep)
            self.writer.add_scalar("tower/pc_loss", torch.mean(pc_loss), timestep)
            self.writer.add_scalar("tower/vr_loss", torch.mean(vr_loss), timestep)
            # self.writer.add_scalar("tower/door_hit", torch.mean(door_hit), timestep)

            lr_scheduler.step()
            self.experience.empty()
            if timestep % 100 == 0:
                name = "ppo" if self.ppo else "a2c"
                path = os.path.join(
                    MODEL_PATH, "model_{}_{}.bin".format(name, timestep)
                )
                torch.save(self.agent_network.state_dict(), path)

        self.writer.close()

    def _lr(self, lr_scheduler):
        return lr_scheduler.get_lr()

    def _fill_experience(self, timestep, action_size):
        counter = 0
        for step in tqdm(range(self.experience_history_size)):
            with torch.no_grad():
                if not timestep:
                    old_state, key, old_time = self.env.reset()
                    last_rhs = self.experience.last_hidden_state
                    reward_action = torch.zeros((self.num_envs, action_size + 1)).to(
                        self.device
                    )
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs
                    )
                    _, q_aux_max = self.agent_network.pixel_control_act(
                        old_state, reward_action, last_rhs
                    )
                else:
                    last_rhs = self.experience.last_hidden_state
                    old_state, reward_action, old_time = self.experience.last_frames()
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs
                    )
                    _, q_aux_max = self.agent_network.pixel_control_act(
                        old_state, reward_action, last_rhs
                    )

                action = self.sample_action(policy_acts)
                new_actions = [self.action_space[act] for act in action]
                policies = policy_acts.view(action_size, self.num_envs)

                self.experience.last_hidden_state = rhs
                new_state, key, new_time, reward, done = self.env.step(new_actions)
                if len(torch.nonzero(reward)) > 0:
                    counter += 1

                action_encoding = torch.zeros((action_size, self.num_envs))
                for i in range(self.num_envs):
                    action_encoding[action[i], i] = 1

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
                    q_aux_max,
                    rhs[0],
                    rhs[1],
                    policies,
                )
                self.experience.increase_frame_pointer()
                timestep += 1
        print("Hits in episode run: {}".format(counter))

    def _update_observations(self, action_size):
        value_loss = torch.zeros(self.num_of_epoches)
        policy_loss = torch.zeros(self.num_of_epoches)
        entropy_loss = torch.zeros(self.num_of_epoches)
        value_replay_loss = torch.zeros(self.num_of_epoches)
        pixel_control_loss = torch.zeros(self.num_of_epoches)
        door_hit = torch.zeros(self.num_of_epoches)

        self.optim.zero_grad()
        for i in tqdm(range(self.num_of_epoches)):
            exp_batches = self.experience.sample_observations(self.sequence_length)
            states, reward_actions, action_indices, policy, rewards, values, _, _, rhs, dones = (
                exp_batches
            )

            agent_loss, pi_loss, base_v_loss, entropy = self.base_loss(
                action_size,
                states,
                reward_actions,
                action_indices,
                rewards,
                values,
                policy,
                rhs,
                dones,
            )

            pc_loss = self.pc_control(action_size)
            v_loss = self.value_replay(action_size)

            loss = agent_loss + v_loss + pc_loss

            value_loss[i].copy_(base_v_loss)
            policy_loss[i].copy_(pi_loss)
            entropy_loss[i].copy_(entropy)
            pixel_control_loss[i].copy_(pc_loss)
            value_replay_loss[i].copy_(v_loss)
            # door_hit[i] = counter

            loss = loss / (i + 1)
            loss.backward()
            if i == self.num_of_epoches - 1:
                torch.nn.utils.clip_grad_norm_(self.agent_network.parameters(), 0.5)
                self.optim.step()

        return (
            policy_loss,
            value_loss,
            entropy_loss,
            pixel_control_loss,
            value_replay_loss,
            door_hit,
        )

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
                rewards[env], values[env], dones[env]
            )
            returns = torch.Tensor(returns).to(self.device)

            adv = returns - values[env]

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

    def pc_control(self, action_size):
        exp_batches = self.experience.sample_observations(self.sequence_length)
        states, reward_actions, action_indices, _, rewards, _, q_auxes, pixel_controls, rhs, dones = (
            exp_batches
        )

        batch_pc_returns = []

        for env in range(self.num_envs):
            pc_returns = self.experience.compute_pc_returns(
                q_auxes[env], rewards[env], pixel_controls[env], dones[env]
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
        states, reward_actions, _, _, rewards, values, _, _, rhs, dones = exp_batches

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
