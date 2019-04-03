import torch


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

    def add_frame(self, frame, env_id):
        # check if frame is second terminal state in a row

        self.memory[self.first_free_frame_index][env_id] = frame
        self.first_free_frame_index += 1


class MemoryFrame:
    def __init__(self, state, key, time, reward, action_encoding, done):
        self.frame = state
        self.key = key
        self.time = time
        self.done_state = done
        self.reward = reward
        self.pixel_change = self._calculate_pixel_change(state)
        self.reward_and_last_action = self._concatenate_reward_and_action(
            reward, action_encoding)

    def isdone_state(self):
        return self.done_state

    def _calculate_pixel_change(self, frame):
        pass

    def _concatenate_reward_and_action(self, reward, action_encoding):
        """
        Concatenate one-hot action representation from last state
        and current state reward.
        """
        return torch.cat((action_encoding, reward), dim=0)
