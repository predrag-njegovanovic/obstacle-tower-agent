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
        return self.distribution(probs=actions).sample()

    def train(self):
        num_of_updates = self.total_timesteps // self.batch_size
        action_size = len(self.action_space)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=lambda step: 1 - (step / float(num_of_updates))
        )

        for timestep in range(num_of_updates):
            self._fill_experience(action_size)
            pi_loss, val_loss, ent, reward, fwd, inv = self._update_observations(
                action_size)
            self.writer.add_scalar("tower/rewards", reward.mean(), timestep)
            self.writer.add_scalar("tower/policy_loss", pi_loss.mean(), timestep)
            self.writer.add_scalar("tower/value_loss", val_loss.mean(), timestep)
            self.writer.add_scalar("tower/entropy_loss", ent.mean(), timestep)
            self.writer.add_scalar("tower/forward_loss", fwd.mean(), timestep)
            self.writer.add_scalar("tower/inverse_loss", inv.mean(), timestep)
            self.writer.add_scalar("tower/lr", np.array(lr_scheduler.get_lr()), timestep)

            lr_scheduler.step()
            self.experience.empty()
            if timestep % 100 == 0:
                name = "ppo" if self.ppo else "a2c"
                path = os.path.join(
                    MODEL_PATH, "model_{}_{}.bin".format(name, timestep)
                )
                torch.save(self.agent_network.state_dict(), path)

        self.writer.close()

    def _fill_experience(self, action_size):
        reset = True

        counter = 0
        for _ in tqdm(range(self.batch_size)):
            with torch.no_grad():
                if reset:
                    old_state, key, old_time = self.env.reset()
                    last_rhs = None
                    reward_action = torch.zeros(
                        (self.num_envs, action_size)).to(self.device)
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs
                    )
                    reset = False
                else:
                    last_rhs = self.experience.last_hidden_state
                    old_state, reward_action = self.experience.last_frames()
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action.cuda(), last_rhs
                    )

                action = self.sample_action(policy_acts)
                new_actions = [self.action_space[act] for act in action]

                self.experience.last_hidden_state = rhs
                new_state, key, new_time, reward, done = self.env.step(new_actions)

                action_encoding = torch.zeros(
                    (self.num_envs, action_size)).to(self.device)
                for i in range(self.num_envs):
                    action_encoding[i, action[i]] = 1

                reward_i, state_features, new_state_features = self.agent_network.icm_act(
                    old_state, new_state, action_encoding)

                reward = torch.clamp(reward, -1, 1)
                reward_e_i = reward + reward_i.cpu()

                if len(torch.nonzero(done)) > 0:
                    reset = True

                if len(torch.nonzero(reward)) > 0:
                    counter += 1

                self.experience.add_experience(
                    new_state,
                    old_state,
                    reward_e_i,
                    action_encoding,
                    done,
                    value,
                    rhs[0],
                    rhs[1],
                    policy_acts,
                    state_features,
                    new_state_features
                )
                self.experience.increase_frame_pointer()

        print("Hits in episode run: {}".format(counter))

    def _update_observations(self, action_size):
        num_updates = self.num_envs // 2

        v_loss_mean = torch.zeros(num_updates)
        pi_loss_mean = torch.zeros(num_updates)
        ent_loss_mean = torch.zeros(num_updates)
        fwd_loss_mean = torch.zeros(num_updates)
        inv_loss_mean = torch.zeros(num_updates)
        rewards_mean = torch.zeros(num_updates)

        for update in range(num_updates):
            value_loss = torch.zeros(self.num_of_epoches)
            policy_loss = torch.zeros(self.num_of_epoches)
            entropy_loss = torch.zeros(self.num_of_epoches)
            fwd_loss = torch.zeros(self.num_of_epoches)
            inv_loss = torch.zeros(self.num_of_epoches)
            rewards_acc = torch.zeros(self.num_of_epoches)

            for i in tqdm(range(self.num_of_epoches)):
                exp_batches = self.experience.on_policy_sampling()

                states, action_indices, policy, rewards, values, rhs, dones, states_f, new_states_f = (
                    exp_batches)
                agent_loss, pi_loss, v_loss, ent, fwd, inv = self.base_loss(action_size,
                                                                            states,
                                                                            action_indices,
                                                                            rewards,
                                                                            values,
                                                                            policy,
                                                                            rhs,
                                                                            dones,
                                                                            states_f,
                                                                            new_states_f)

                self.optim.zero_grad()
                loss = agent_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.agent_network.parameters(), 0.5)
                self.optim.step()

                value_loss[i].copy_(v_loss)
                policy_loss[i].copy_(pi_loss)
                entropy_loss[i].copy_(ent)
                rewards_acc[i].copy_(torch.cat(rewards).mean())
                fwd_loss[i].copy_(fwd)
                inv_loss[i].copy_(inv)

            v_loss_mean[update].copy_(value_loss.mean())
            pi_loss_mean[update].copy_(policy_loss.mean())
            ent_loss_mean[update].copy_(entropy_loss.mean())
            rewards_mean[update].copy_(rewards_acc.mean())
            fwd_loss_mean[update].copy_(fwd_loss.mean())
            inv_loss_mean[update].copy_(inv_loss.mean())

        return (
            pi_loss_mean,
            v_loss_mean,
            ent_loss_mean,
            rewards_mean,
            fwd_loss_mean,
            inv_loss_mean,
        )

    def base_loss(
        self,
        action_size,
        states,
        action_indices,
        rewards,
        values,
        old_policy,
        base_rhs,
        dones,
        state_features,
        new_state_features
    ):
        batch_returns = []
        batch_advantages = []

        for env in range(2):
            returns = self.experience.compute_returns(
                rewards[env], values[env], dones[env]
            )
            returns = torch.Tensor(returns).to(self.device)

            adv = returns - values[env]

            batch_advantages.append(adv)
            batch_returns.append(returns)

        advantage = torch.cat(batch_advantages, dim=0)
        advantage = (advantage - torch.mean(advantage, dim=0)) / \
            torch.std(advantage) + 1e-6

        new_value, policy_acts, _ = self.agent_network.act(
            states, action_indices, base_rhs
        )
        returns = torch.cat(batch_returns, dim=0).to(self.device)

        batch_predicted_states = self.agent_network.forward_act(
            state_features, action_indices)
        batch_predicted_acts = self.agent_network.inverse_act(
            state_features, new_state_features)

        if self.ppo:
            agent_loss, pi_loss, v_loss, entropy = self.agent_network.ppo_loss(
                old_policy, policy_acts, advantage, returns, new_value, action_indices
            )
        else:
            agent_loss, pi_loss, v_loss, entropy, fwd_loss, inv_loss = self.agent_network.a2c_loss(
                policy_acts, advantage, returns, new_value, action_indices,
                new_state_features, batch_predicted_states, batch_predicted_acts)

        return agent_loss, pi_loss, v_loss, entropy, fwd_loss, inv_loss
