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
            min_reward, max_reward, std_reward, mean_reward = (
                self.experience.reward_stats()
            )

            self.writer.add_scalars(
                "tower/rewards",
                {
                    "mean": mean_reward,
                    "min": min_reward,
                    "max": max_reward,
                    "std": std_reward,
                },
                timestep,
            )
            self.writer.add_scalar("tower/policy_loss", torch.mean(pi_loss), timestep)
            self.writer.add_scalar("tower/value_loss", torch.mean(v_loss), timestep)
            self.writer.add_scalar("tower/entropy_loss", torch.mean(entropy), timestep)
            self.writer.add_scalar("tower/pc_loss", torch.mean(pc_loss), timestep)
            self.writer.add_scalar("tower/vr_loss", torch.mean(vr_loss), timestep)
            self.writer.add_scalar("tower/door_hit", torch.mean(door_hit), timestep)

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
        for step in tqdm(range(self.experience_history_size)):
            with torch.no_grad():
                if not step:
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

                self.experience.last_hidden_state = rhs
                new_state, key, new_time, reward, done = self.env.step(new_actions)

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
                )
                self.experience.increase_frame_pointer()

    def _update_observations(self, action_size):
        epoches = self.num_of_epoches

        value_loss = torch.zeros(epoches)
        policy_loss = torch.zeros(epoches)
        entropy_loss = torch.zeros(epoches)
        value_replay_loss = torch.zeros(epoches)
        pixel_control_loss = torch.zeros(epoches)
        door_hit = torch.zeros(epoches)

        for i in tqdm(range(0, epoches)):
            states = []
            values = []
            policies = []
            rewards = []
            reward_actions = []
            action_indices = []
            counter = 0

            last_rhs = None
            old_state, _, old_time = self.env.reset()
            reward_action = torch.zeros((self.num_envs, action_size + 1)).to(
                self.device
            )
            for _ in range(self.batch_size):
                with torch.no_grad():
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs
                    )

                    action = self.sample_action(policy_acts)
                    new_actions = [self.action_space[act] for act in action]

                    new_state, key, new_time, reward, done = self.env.step(new_actions)
                    if len(torch.nonzero(reward)):
                        counter += 1
                    if len(torch.nonzero(done)) > 0:
                        print("Here at i = {}".format(i))
                        break

                    action_encoding = torch.zeros((action_size, self.num_envs))
                    pi_acts = policy_acts.view(action_size, self.num_envs)
                    for ind in range(self.num_envs):
                        action_encoding[action[ind], ind] = 1

                    calc_reward = self.experience.calculate_reward(
                        reward, new_time, old_time, key
                    )
                    new_reward_action = (
                        self.experience.concatenate_reward_and_action(
                            calc_reward, action_encoding
                        )
                        .transpose_(1, 0)
                        .to(self.device)
                    )

                    last_rhs = rhs
                    old_state = new_state
                    old_time = new_time
                    reward_action = new_reward_action
                    pi_acts = torch.mul(pi_acts, action_encoding.cuda())

                    pi_acts.transpose_(1, 0)
                    action_encoding.transpose_(1, 0)

                    states.append(new_state)
                    values.append(value)
                    policies.append(pi_acts)
                    action_indices.append(action_encoding)
                    rewards.append(calc_reward)
                    reward_actions.append(new_reward_action)

            if not len(states):
                continue

            state_tensor = torch.cat(states, dim=0)
            reward_actions_tensor = torch.cat(reward_actions, dim=0)
            action_indices_tensor = torch.cat(action_indices, dim=0).to(self.device)
            policy_tensor = torch.cat(policies, dim=0)
            rewards = torch.stack(rewards)
            values = torch.stack(values)

            agent_loss, pi_loss, v_loss, entropy = self.base_loss(
                action_size,
                state_tensor,
                reward_actions_tensor,
                action_indices_tensor,
                rewards,
                values,
                policy_tensor,
            )
            pc_loss = self.pc_control(action_size)
            v_loss = self.value_replay(action_size)

            loss = agent_loss + v_loss + pc_loss

            value_loss[i].copy_(v_loss)
            policy_loss[i].copy_(pi_loss)
            entropy_loss[i].copy_(entropy)
            pixel_control_loss[i].copy_(pc_loss)
            value_replay_loss[i].copy_(v_loss)
            door_hit[i] = counter

            print("Epoche: {} and number of passings: {}".format(i, counter))
            print(
                "Epoche: {} and sum rewards in all envs is: {}".format(
                    i, torch.sum(rewards, dim=0)
                )
            )

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent_network.parameters(), 10)
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
    ):
        batch_returns = []
        batch_advantages = []

        for env in range(self.num_envs):
            returns = self.experience.compute_returns(
                rewards[:, env].cpu().numpy(), values[:, env].cpu().numpy()
            )
            returns = torch.Tensor(returns).to(self.device)

            adv = returns - values[:, env]

            batch_advantages.append(adv)
            batch_returns.append(returns)

        advantage = torch.cat(batch_advantages, dim=0).unsqueeze(-1)
        advantage = (advantage - torch.mean(advantage, dim=0)) / torch.std(
            advantage + 1e-6
        )

        new_value, policy_acts, _ = self.agent_network.act(states, reward_actions)
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
        exp_batches = self.experience.sample_observations(128)
        states, reward_actions, action_indices, rewards, _, q_auxes, pixel_controls = (
            exp_batches
        )

        batch_pc_returns = []

        for env in range(self.num_envs):
            pc_returns = self.experience.compute_pc_returns(
                q_auxes[env], rewards[env], pixel_controls[env]
            )
            batch_pc_returns.append(pc_returns)

        pc_returns = torch.cat(batch_pc_returns, dim=0).to(self.device)

        new_value, policy_acts, _ = self.agent_network.act(states, reward_actions)
        q_aux, _ = self.agent_network.pixel_control_act(states, reward_actions)

        pc_loss = self.agent_network.pc_loss(
            action_size, action_indices, q_aux, pc_returns
        )

        return pc_loss

    def value_replay(self, action_size):
        exp_batches = self.experience.sample_observations(128)
        states, reward_actions, _, rewards, values, _, _ = exp_batches

        batch_v_returns = []

        for env in range(self.num_envs):
            v_returns = self.experience.compute_v_returns(rewards[env], values[env])
            v_returns = torch.Tensor(v_returns).to(self.device)
            batch_v_returns.append(v_returns)

        new_value, _, _ = self.agent_network.act(states, reward_actions)

        v_returns = torch.cat(batch_v_returns, dim=0).to(self.device)
        v_loss = self.agent_network.v_loss(v_returns, new_value)

        return v_loss
