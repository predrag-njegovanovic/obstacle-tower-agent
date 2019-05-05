import torch
import random
import numpy as np


class ExperienceMemory:
    def __init__(self, num_envs, memory_size, action_size, device):
        self._init_memory(num_envs, memory_size, action_size, device)
        self.memory_size = memory_size
        self.action_size = action_size
        self.num_envs = num_envs
        self.memory_pointer = 0
        self._last_hidden_state = None
        self.device = device

    @property
    def last_hidden_state(self):
        return self._last_hidden_state

    @last_hidden_state.setter
    def last_hidden_state(self, value):
        self._last_hidden_state = value

    def _init_memory(self, num_envs, memory_size, action_size, device):
        self.frame = (
            torch.zeros((memory_size, num_envs, 3, 84, 84)).type(torch.uint8).to(device)
        )
        self.reward = torch.zeros((memory_size, num_envs))
        self.done_state = torch.zeros((memory_size, num_envs))
        self.value = torch.zeros((memory_size, num_envs)).to(device)
        self.action_indices = torch.zeros((memory_size, num_envs, action_size)).to(
            device
        )
        self.reward_action = torch.zeros((memory_size, num_envs, action_size + 1)).to(
            device
        )
        self.first_lstm_rhs = torch.zeros((memory_size, 1, num_envs, 256)).to(device)
        self.second_lstm_rhs = torch.zeros((memory_size, 1, num_envs, 256)).to(device)
        self.policy_values = torch.zeros((memory_size, num_envs, action_size)).to(
            device
        )
        self.state_f = torch.zeros((memory_size, num_envs, 288)).to(device)
        self.new_state_f = torch.zeros((memory_size, num_envs, 288)).to(device)

    def empty(self):
        self.frame[0].copy_(self.frame[-1])
        self.reward_action[0].copy_(self.reward_action[-1])
        self.memory_pointer = 0

    def add_experience(
        self,
        new_state,
        old_state,
        reward,
        action_encoding,
        done,
        predicted_value,
        rhs_first,
        rhs_second,
        policy,
        state_f,
        new_state_f
    ):

        self.frame[self.memory_pointer].copy_(new_state)
        self.reward[self.memory_pointer].copy_(reward)
        self.value[self.memory_pointer].copy_(predicted_value)
        self.action_indices[self.memory_pointer].copy_(action_encoding)
        self.reward_action[self.memory_pointer].copy_(
            self.concatenate_reward_and_action(reward, action_encoding)
        )
        self.done_state[self.memory_pointer].copy_(done)
        self.first_lstm_rhs[self.memory_pointer].copy_(rhs_first)
        self.second_lstm_rhs[self.memory_pointer].copy_(rhs_second)
        self.policy_values[self.memory_pointer].copy_(policy)
        self.state_f[self.memory_pointer].copy_(state_f)
        self.new_state_f[self.memory_pointer].copy_(new_state_f)

    def increase_frame_pointer(self):
        self.memory_pointer += 1

    def last_frames(self):
        states = self.frame[self.memory_pointer - 1]
        reward_actions = self.reward_action[self.memory_pointer - 1]

        return states, reward_actions

    def on_policy_sampling(self):
        batched_value = []
        batched_states = []
        batched_reward = []
        batched_action_indices = []
        batched_first_rhs = []
        batched_second_rhs = []
        batched_policy = []
        batched_dones = []
        batched_state_f = []
        batched_new_state_f = []

        rand_envs = torch.randperm(self.num_envs)

        for i in range(0, self.num_envs, self.num_envs // 2):
            env = rand_envs[i]

            states = self.frame[:, env, :, :, :]
            rewards = self.reward[:, env]
            values = self.value[:, env]
            action_indices = self.action_indices[:, env, :]
            first_rhs = self.first_lstm_rhs[:, :, env, :]
            second_rhs = self.second_lstm_rhs[:, :, env, :]
            policies = self.policy_values[:, env, :]
            dones = self.done_state[:, env]
            states_f = self.state_f[:, env, :]
            new_states_f = self.new_state_f[:, env, :]

            batched_value.append(values)
            batched_states.append(states)
            batched_reward.append(rewards)
            batched_action_indices.append(action_indices)
            batched_first_rhs.append(first_rhs)
            batched_second_rhs.append(second_rhs)
            batched_policy.append(policies)
            batched_dones.append(dones)
            batched_state_f.append(states_f)
            batched_new_state_f.append(new_states_f)

        return (
            torch.cat(batched_states, dim=0),
            torch.cat(batched_action_indices, dim=0),
            torch.cat(batched_policy, dim=0),
            batched_reward,
            batched_value,
            (
                torch.cat(batched_first_rhs, dim=0).view(1, -1, 256),
                torch.cat(batched_second_rhs, dim=0).view(1, -1, 256),
            ),
            batched_dones,
            torch.cat(batched_state_f, dim=0),
            torch.cat(batched_new_state_f, dim=0)
        )

    def sample_observations(self, sequence):
        batched_value = []
        batched_q_aux = []
        batched_states = []
        batched_reward = []
        batched_pixel_control = []
        batched_action_indices = []
        batched_reward_actions = []
        batched_first_rhs = []
        batched_second_rhs = []
        batched_policy = []
        batched_dones = []

        for env in range(self.num_envs):
            start = random.randint(0, self.memory_size - sequence - 2)
            sample_index = start + sequence

            if self.done_state[start, env] or self.done_state[start + 1, env]:
                start += 1
                if self.done_state[start, env]:
                    start += 1

            for i in range(sequence):
                if self.done_state[start + i, env]:
                    sample_index = start + i - 1
                    break

            if sample_index <= start:
                continue

            states = self.frame[start:sample_index, env, :, :, :]
            reward_actions = self.reward_action[start:sample_index, :, env]
            rewards = self.reward[start:sample_index, env]
            values = self.value[start:sample_index, env]
            pixel_controls = self.pixel_change[start:sample_index, env, :, :]
            action_indices = self.action_indices[start:sample_index, :, env]
            q_auxes = self.q_aux[start:sample_index, env, :, :]
            first_rhs = self.first_lstm_rhs[start:sample_index, :, env, :]
            second_rhs = self.second_lstm_rhs[start:sample_index, :, env, :]
            policies = self.policy_values[start:sample_index, :, env]
            dones = self.done_state[start:sample_index, env]

            batched_value.append(values)
            batched_q_aux.append(q_auxes)
            batched_states.append(states)
            batched_reward.append(rewards)
            batched_pixel_control.append(pixel_controls)
            batched_reward_actions.append(reward_actions)
            batched_action_indices.append(action_indices)
            batched_first_rhs.append(first_rhs)
            batched_second_rhs.append(second_rhs)
            batched_policy.append(policies)
            batched_dones.append(dones)

        return (
            torch.cat(batched_states, dim=0),
            torch.cat(batched_reward_actions, dim=0),
            torch.cat(batched_action_indices, dim=0),
            torch.cat(batched_policy, dim=0),
            batched_reward,
            batched_value,
            batched_q_aux,
            batched_pixel_control,
            (
                torch.cat(batched_first_rhs, dim=0).view(1, -1, 256),
                torch.cat(batched_second_rhs, dim=0).view(1, -1, 256),
            ),
            batched_dones,
        )

    # try gae later
    def compute_returns(self, rewards, values, dones, discount=0.99):
        num_steps = rewards.shape[0]

        returns = np.zeros((num_steps))
        if not dones[-1]:
            returns[-1] = values[-1]

        for step in reversed(range(num_steps - 1)):
            returns[step] = rewards[step] + discount * returns[step + 1]

        return returns

    def compute_pc_returns(self, q_aux, rewards, pixel_controls, dones, gamma=0.9):
        num_steps, height, width = pixel_controls.shape

        pc_returns = torch.zeros((num_steps, height, width)).to(self.device)
        if not dones[-1]:
            pc_returns[-1] = q_aux

        for step in reversed(range(num_steps - 1)):
            pc_returns[step] = (
                pixel_controls[step] + gamma * pc_returns[step + 1]
            )

        return pc_returns

    def compute_v_returns(self, rewards, values, dones, gamma=1.0):
        num_steps = values.shape[0]

        v_returns = np.zeros((num_steps))
        if not dones[-1]:
            v_returns[-1] = values[-1]

        for step in reversed(range(num_steps - 1)):
            v_returns[step] = rewards[step] + gamma * v_returns[step + 1]

        return v_returns

    def calculate_reward(self, reward, new_time, old_time, key):
        """
        Scale time difference between two steps to [0, 1] range and add to reward.
        """
        diff = torch.zeros(new_time.shape)
        for index, _ in enumerate(reward):
            if reward[index] == 0.1:
                reward[index] *= 10

            time_difference = new_time[index] - old_time[index]
            diff[index] = time_difference / 1000

        return reward + diff + 0.2 * key

    def _subsample(self, frame_mean_diff, piece_size=4):
        shapes = frame_mean_diff.shape
        subsamples_shape = (
            self.num_envs,
            shapes[1] // piece_size,
            piece_size,
            shapes[2] // piece_size,
            piece_size,
        )
        reshaped_frame = frame_mean_diff.reshape(subsamples_shape)
        reshaped_mean = torch.mean(reshaped_frame, -1)
        pc_output = torch.mean(reshaped_mean, 2)
        return pc_output

    def _calculate_pixel_change(self, new_state, old_state):
        """
        Calculate pixel change between two states by creating
        frame differences and then subsampling it to twenty 4x4 pieces.
        """
        new_state = new_state.type(torch.float32) / 255
        old_state = old_state.type(torch.float32) / 255

        frame_diff = torch.abs(
            new_state[:, :, 2:-2, 2:-2] - old_state[:, :, 2:-2, 2:-2]
        )
        frame_mean = torch.mean(frame_diff, dim=1, keepdim=False)
        subsampled_frame = self._subsample(frame_mean)
        return subsampled_frame

    def concatenate_reward_and_action(self, reward, action_encoding):
        """
        Concatenate one-hot action representation from last state
        and current state reward.
        """
        return torch.cat((action_encoding.cpu(), reward.unsqueeze(1)), dim=1).to(self.device)
