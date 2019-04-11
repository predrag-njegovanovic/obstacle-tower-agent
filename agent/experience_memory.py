import torch
import random
import numpy as np

from agent.utils import torch_device


class ExperienceMemory:
    def __init__(self, num_envs, memory_size, action_size):
        self._init_memory(num_envs, memory_size, action_size)
        self.memory_size = memory_size
        self.action_size = action_size
        self.num_envs = num_envs
        self.memory_pointer = 0
        self._last_hidden_state = None

    @property
    def last_hidden_state(self):
        return self._last_hidden_state

    @last_hidden_state.setter
    def last_hidden_state(self, value):
        self._last_hidden_state = value

    def _init_memory(self, num_envs, memory_size, action_size):
        self.frame = (
            torch.zeros((memory_size, num_envs, 3, 84, 84))
            .type(torch.uint8)
            .to(torch_device())
        )
        self.time = torch.zeros((memory_size, num_envs))
        self.key = torch.zeros((memory_size, num_envs))
        self.reward = torch.zeros((memory_size, num_envs))
        self.done_state = torch.zeros((memory_size, num_envs))
        self.policy_values = torch.zeros((memory_size, action_size, num_envs)).to(
            torch_device()
        )
        self.value = torch.zeros((memory_size, num_envs)).to(torch_device())
        self.pixel_change = torch.zeros((memory_size, num_envs, 20, 20)).type(
            torch.uint8
        )
        self.action_indices = torch.zeros((memory_size, action_size, num_envs)).to(
            torch_device()
        )
        self.reward_action = torch.zeros((memory_size, action_size + 1, num_envs)).to(
            torch_device()
        )

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

    def increase_frame_pointer(self):
        self.memory_pointer += 1

    def last_frames(self):
        states = self.frame[self.memory_pointer - 1]
        time = self.time[self.memory_pointer - 1]
        reward_actions = self.reward_action[self.memory_pointer - 1]

        return states, reward_actions.view(self.num_envs, self.action_size + 1), time

    def sample_observations(self, sequence_size):
        batched_reward = []
        batched_value = []
        batched_policy = []
        batched_pixel_control = []
        batched_action_indices = []

        for env in range(self.num_envs):
            done_flag = False
            start = random.randint(0, self.memory_size - sequence_size - 1)

            # If terminate state, start from next one
            if self.done_state[start, env]:
                start += 1

            for i in range(sequence_size):
                if self.done_state[start + i, env]:
                    rewards = self.reward[start : start + i - 1, env]
                    values = self.value[start : start + i - 1, env]
                    policy_values = self.policy_values[start : start + i - 1, :, env]
                    pixel_controls = self.pixel_change[start : start + i - 1, env, :, :]
                    action_indices = self.action_indices[start : start + i - 1, env]
                    done_flag = True
                    break

            if not done_flag:
                rewards = self.reward[start : start + sequence_size, env]
                values = self.value[start : start + sequence_size, env]
                policy_values = self.policy_values[
                    start : start + sequence_size, :, env
                ]
                pixel_controls = self.pixel_change[
                    start : start + sequence_size, env, :, :
                ]
                action_indices = self.action_indices[start : start + sequence_size, env]

            batched_reward.append(rewards)
            batched_value.append(values)
            batched_policy.append(policy_values)
            batched_pixel_control.append(pixel_controls)
            batched_action_indices.append(action_indices)

        return (
            batched_reward,
            batched_value,
            torch.cat(batched_policy, dim=0),
            batched_pixel_control,
            torch.cat(batched_action_indices, dim=0),
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

    def _calculate_reward(self, reward, new_time, old_time, key):
        return reward + self._time_normalize(new_time, old_time) + 0.2 * key

    def _time_normalize(self, new_time, old_time):
        """
        Scale time difference between two steps to [0, 1] range.
        """
        difference = new_time - old_time
        return difference / 1000

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
        action_encoding[-1] = reward
        return action_encoding
