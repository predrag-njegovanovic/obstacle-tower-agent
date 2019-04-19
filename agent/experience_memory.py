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
        self.time = torch.zeros((memory_size, num_envs))
        self.key = torch.zeros((memory_size, num_envs))
        self.reward = torch.zeros((memory_size, num_envs))
        self.done_state = torch.zeros((memory_size, num_envs))
        self.policy_values = torch.zeros((memory_size, action_size, num_envs)).to(
            device
        )
        self.value = torch.zeros((memory_size, num_envs)).to(device)
        self.pixel_change = torch.zeros((memory_size, num_envs, 20, 20)).type(
            torch.uint8
        )
        self.q_aux = torch.zeros((memory_size, num_envs, 20, 20))
        self.action_indices = torch.zeros((memory_size, action_size, num_envs)).to(
            device
        )
        self.reward_action = torch.zeros((memory_size, action_size + 1, num_envs)).to(
            device
        )

    def mean_reward(self):
        return torch.mean(self.reward).item()

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
        policy_value,
        q_aux,
    ):

        self.frame[self.memory_pointer].copy_(new_state)
        self.time[self.memory_pointer].copy_(new_time)
        self.key[self.memory_pointer].copy_(key)
        self.reward[self.memory_pointer].copy_(
            self._calculate_reward(reward, new_time, old_time, key)
        )
        self.value[self.memory_pointer].copy_(predicted_value)
        self.policy_values[self.memory_pointer].copy_(policy_value)
        self.action_indices[self.memory_pointer].copy_(action_encoding)
        self.reward_action[self.memory_pointer].copy_(
            self._concatenate_reward_and_action(reward, action_encoding)
        )
        self.done_state[self.memory_pointer].copy_(done)
        self.pixel_change[self.memory_pointer].copy_(
            self._calculate_pixel_change(new_state, old_state)
        )
        self.q_aux[self.memory_pointer].copy_(q_aux)

    def increase_frame_pointer(self):
        self.memory_pointer += 1

    def last_frames(self):
        states = self.frame[self.memory_pointer - 1]
        time = self.time[self.memory_pointer - 1]
        reward_actions = self.reward_action[self.memory_pointer - 1]

        return states, reward_actions.view(self.num_envs, self.action_size + 1), time

    def sample_observations(self, sequence):
        batched_value = []
        batched_q_aux = []
        batched_policy = []
        batched_states = []
        batched_reward = []
        batched_pixel_control = []
        batched_action_indices = []
        batched_reward_actions = []

        for env in range(self.num_envs):
            sample_index = 0
            start = random.randint(0, self.memory_size - sequence - 1)

            if self.done_state[start, env]:
                start += 1

            for i in range(sequence):
                if self.done_state[start + i, env]:
                    sample_index = start + i - 1
                    break

            sample_index = sequence

            if sample_index <= start:
                continue

            states = self.frame[start:sample_index, env, :, :, :]
            reward_actions = self.reward_action[start:sample_index, :, env]
            rewards = self.reward[start:sample_index, env]
            values = self.value[start:sample_index, env]
            pixel_controls = self.pixel_change[start:sample_index, env, :, :]
            action_indices = self.action_indices[start:sample_index, :, env]
            q_auxes = self.q_aux[start:sample_index, env, :, :]
            policies = self.policy_values[start:sample_index, :, env]

            batched_value.append(values)
            batched_q_aux.append(q_auxes)
            batched_states.append(states)
            batched_reward.append(rewards)
            batched_policy.append(policies)
            batched_pixel_control.append(pixel_controls)
            batched_reward_actions.append(reward_actions)
            batched_action_indices.append(action_indices)

        return (
            torch.cat(batched_states, dim=0),
            torch.cat(batched_reward_actions, dim=0),
            torch.cat(batched_action_indices, dim=0),
            torch.cat(batched_policy, dim=0),
            batched_reward,
            batched_value,
            batched_q_aux,
            batched_pixel_control,
        )

    # try gae later
    def compute_returns(self, rewards, values, discount=0.99):
        num_steps = rewards.shape[0]

        returns = np.zeros((num_steps))
        if rewards[-1]:
            returns[-1] = rewards[-1]
        else:
            returns[-1] = values[-1]

        for step in reversed(range(num_steps - 1)):
            returns[step] = rewards[step] + discount * returns[step + 1]

        return returns

    def compute_pc_returns(self, q_aux, rewards, pixel_controls, gamma=0.9):
        num_steps, height, width = pixel_controls.shape

        pc_returns = torch.zeros((num_steps, height, width))
        if rewards[-1]:
            pc_returns[-1] = q_aux[-1]

        for step in reversed(range(num_steps - 1)):
            pc_returns[step] = (
                pixel_controls[step].type(torch.float32) + gamma * pc_returns[step + 1]
            )

        return pc_returns

    def compute_v_returns(self, rewards, values, gamma=1.0):
        num_steps = values.shape[0]

        v_returns = np.zeros((num_steps))
        if rewards[-1]:
            v_returns[-1] = values[-1]

        for step in reversed(range(num_steps - 1)):
            v_returns[step] = rewards[step] + gamma * v_returns[step + 1]

        return v_returns

    def _calculate_reward(self, reward, new_time, old_time, key):
        return reward + self._time_normalize(reward, new_time, old_time) + 0.2 * key

    def _time_normalize(self, reward, new_time, old_time):
        """
        Scale time difference between two steps to [0, 1] range.
        """
        difference = new_time - old_time
        if reward:
            return difference / 1000
        else:
            return difference / 10000

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
        new_state = new_state.type(torch.float32)
        old_state = old_state.type(torch.float32)

        frame_diff = torch.abs(
            new_state[:, :, 2:-2, 2:-2] - old_state[:, :, 2:-2, 2:-2]
        )
        frame_mean = torch.mean(frame_diff, dim=1, keepdim=False)
        subsampled_frame = self._subsample(frame_mean)
        return subsampled_frame

    def _concatenate_reward_and_action(self, reward, action_encoding):
        """
        Concatenate one-hot action representation from last state
        and current state reward.
        """
        clipped_reward = torch.clamp(reward, 1, 0)
        return torch.cat((action_encoding, clipped_reward.unsqueeze(0)), dim=0)
