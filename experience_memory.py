import torch
import random
import numpy as np


class ExperienceMemory:
    def __init__(self, num_envs, memory_size):
        self.memory = torch.zeros((memory_size, num_envs))
        self.full = False
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.first_free_frame_index = 0

    def empty_memory(self):
        self.memory = torch.zeros((self.num_envs, self.memory_size), dtype=object)
        self.full = True
        self.first_free_frame_index = 0

    def add_frame(self, frame, env_id):
        self.memory[self.first_free_frame_index][env_id] = frame
        self.first_free_frame_index += 1

    def last_state(self, env_id):
        return self.memory[self.first_free_frame_index - 1][env_id]

    def sample_frames(self, sequence_size):
        env_indices = list(self.num_envs)
        accumulated_frames = []
        batched_frames = []

        for env in range(self.num_envs):
            frames = []
            start_index = random.randint(0, self.memory_size - sequence_size - 1)

            # If terminate state, start from next one
            if self.memory[start_index][env].done:
                start_index += 1

            for index in range(sequence_size):
                frame = self.memory_size[start_index + index][env]
                if frame.isdone_state():
                    break

                frames.append(frame)
            accumulated_frames.append(frames)

        truncate_size = min([len(frames) for frames in accumulated_frames])

        for env_index in random.shuffle(env_indices):
            batched_frames.append(accumulated_frames[env_index][:truncate_size])

        return torch.Tensor(batched_frames)


class MemoryFrame:
    def __init__(self, new_state, key, time, reward, action_encoding, done, old_state):
        self.frame = new_state
        self.key = key
        self.time = time
        self.done_state = done
        self.reward = reward
        self.pixel_change = self._calculate_pixel_change(new_state, old_state)
        self.reward_and_last_action = self._concatenate_reward_and_action(
            reward, action_encoding)

    def isdone_state(self):
        return self.done_state

    def _subsample(self, frame_mean_diff, piece_size=4):
        shapes = frame_mean_diff.shape
        subsamples_shape = shapes[0] // piece_size, piece_size, shapes[1] // piece_size, piece_size
        return frame_mean_diff.reshape(subsamples_shape).mean(-1).mean(1)

    def _calculate_pixel_change(self, new_state, old_state):
        """
        Calculate pixel change between two states by creating
        frame differences and then subsampling it to twenty 4x4 pieces.
        """
        frame_diff = np.absolute(new_state[:, 2:-2, 2:-2] - old_state[:, 2:-2, 2:-2])
        frame_mean = np.mean(frame_diff, 0)
        subsampled_frame = self._subsample(frame_mean)
        return subsampled_frame

    def _concatenate_reward_and_action(self, reward, action_encoding):
        """
        Concatenate one-hot action representation from last state
        and current state reward.
        """
        return torch.cat((action_encoding, reward), dim=0)
