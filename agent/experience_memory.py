import torch
import random
import itertools
import numpy as np

from collections import deque


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
        self.frame = deque(maxlen=memory_size)
        self.time = deque(maxlen=memory_size)
        self.key = deque(maxlen=memory_size)
        self.reward = deque(maxlen=memory_size)
        self.done_state = deque(maxlen=memory_size)
        self.value = deque(maxlen=memory_size)
        self.pixel_change = deque(maxlen=memory_size)
        self.action_indices = deque(maxlen=memory_size)
        self.reward_action = deque(maxlen=memory_size)
        self.first_lstm_rhs = deque(maxlen=memory_size)
        self.second_lstm_rhs = deque(maxlen=memory_size)
        self.policy_values = deque(maxlen=memory_size)

    def reward_stats(self):
        mean = torch.mean(self.reward)
        return mean

    def empty(self):
        self.frame[0].copy_(self.frame[-1])
        self.time[0].copy_(self.time[-1])
        self.reward_action[0].copy_(self.reward_action[-1])
        self.memory_pointer = 0

    def add_experience(
        self,
        new_state,
        old_state,
        new_time,
        old_time,
        key,
        reward,
        action_encoding,
        done,
        predicted_value,
        rhs_first,
        rhs_second,
        policy,
    ):

        if self.full():
            self.frame.popleft()
            self.time.popleft()
            self.key.popleft()
            self.reward.popleft()
            self.value.popleft()
            self.action_indices.popleft()
            self.reward_action.popleft()
            self.done_state.popleft()
            self.pixel_change.popleft()
            self.policy_values.popleft()
            self.first_lstm_rhs.popleft()
            self.second_lstm_rhs.popleft()

        self.frame.append(new_state)
        self.time.append(new_time)
        self.key.append(key)
        self.reward.append(self.calculate_reward(reward, new_time, old_time, key))
        self.value.append(predicted_value)
        self.action_indices.append(action_encoding)
        self.reward_action.append(
            self.concatenate_reward_and_action(reward, action_encoding))
        self.done_state.append(done)
        self.pixel_change.append(self._calculate_pixel_change(new_state, old_state))
        self.first_lstm_rhs.append(rhs_first)
        self.second_lstm_rhs.append(rhs_second)
        self.policy_values.append(policy)

    def full(self):
        return len(self.frame) == self.memory_size

    def increase_frame_pointer(self):
        self.memory_pointer += 1

    def last_frames(self):
        states = self.frame[-1]
        time = self.time[-1]
        reward_actions = self.reward_action[-1]

        return states, reward_actions.view(self.num_envs, self.action_size + 1), time

    def sample_observations(self, sequence):
        batched_value = []
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

            if self.done_state[start][:, env] or self.done_state[start + 1][:, env]:
                start += 1
                if self.done_state[start][:, env]:
                    start += 1

            for i in range(sequence):
                if self.done_state[start + i, env]:
                    sample_index = start + i - 1
                    break

            if sample_index <= start:
                continue

            # Fix sampling
            states = [frame[env, :, :, :] for frame in self.frame[start:sample_index]]
            reward_actions = [reward_action[:, env]
                              for reward_action in self.reward_action[start:sample_index]]
            rewards = [reward[env] for reward in self.reward[start:sample_index]]
            values = [value[env] for value in self.value[start:sample_index]]
            pixel_controls = [pixel_change[env, :, :]
                              for pixel_change in self.pixel_change[start:sample_index]]
            action_indices = [action_indices[:, env]
                              for action_indices in self.action_indices[start:sample_index]]
            first_rhs = [rhs[:, env, :]
                         for rhs in self.first_lstm_rhs[start:sample_index]]
            second_rhs = [rhs[:, env, :]
                          for rhs in self.second_lstm_rhs[start:sample_index]]
            policies = [policies[:, env]
                        for policies in self.policy_values[start:sample_index]]
            dones = [dones[env] for dones in self.done_state[start:sample_index]]

            batched_value.append(values)
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
            batched_pixel_control,
            (
                torch.cat(batched_first_rhs, dim=0).view(2, -1, 256),
                torch.cat(batched_second_rhs, dim=0).view(2, -1, 256),
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

        pc_returns = torch.zeros((num_steps, height, width))
        if not dones[-1]:
            pc_returns[-1] = q_aux[-1]

        for step in reversed(range(num_steps - 1)):
            pc_returns[step] = (
                pixel_controls[step].type(torch.float32) + gamma * pc_returns[step + 1]
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
            if time_difference <= 0:
                diff[index] = 0
            else:
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
        return pc_output.type(torch.uint8)

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
        clipped_reward = torch.clamp(reward, 0, 1)
        return torch.cat((action_encoding, clipped_reward.unsqueeze(0)), dim=0).to(self.device)
