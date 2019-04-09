import torch

from tqdm import tqdm

from agent.experience_memory import MemoryFrame
from agent.utils import torch_device


class Trainer:
    def __init__(self,
                 parallel_environment,
                 experience_memory,
                 agent_network,
                 action_space,
                 num_envs,
                 experience_history_size,
                 batch_size,
                 num_of_epoches,
                 total_timesteps):

        self.env = parallel_environment
        self.experience_memory = experience_memory
        self.agent_network = agent_network
        self.action_space = action_space
        self.num_envs = num_envs
        self.experience_history_size = experience_history_size
        self.batch_size = batch_size
        self.num_of_epoches = num_of_epoches
        self.total_timesteps = total_timesteps
        self.distribution = torch.distributions.Categorical

    def sample_action(self, actions):
        """
        Sample action from tensor of action probabilities or log probabilities.
        Input: torch.Tensor([8, 54])
        Return: torch.Tensor([8])
        """
        prob_distribution = self.distribution(actions)
        return prob_distribution.sample()

    def train(self):
        self.agent_network.to_cuda()
        for _ in range(0, self.total_timesteps, self.experience_history_size):
            self._fill_experience(len(self.action_space))
            # update network
            self.experience_memory.empty()

    def _fill_experience(self, action_size):
        old_state, key, old_time = self.env.reset()
        for step in tqdm(range(self.experience_history_size)):
            print(step)
            if step == 0:
                reward_action = torch.zeros(
                    (self.num_envs, action_size + 1)).to(torch_device())
                value, policy_acts, rhs = self.agent_network.act(old_state, reward_action)
            else:
                last_rhs = self.experience_memory.last_hidden_state
                old_state, reward_action, old_time = self.experience_memory.last_frames()
                value, policy_acts, rhs = self.agent_network.act(
                    old_state, reward_action, last_rhs)

            action = self.sample_action(policy_acts)
            new_actions = [self.action_space[act] for act in action]
            self.experience_memory.last_hidden_state = rhs
            new_state, key, new_time, reward, done = self.env.step(new_actions)

            for i in range(self.num_envs):
                action_encoding = torch.zeros((action_size + 1))
                action_encoding[action[i]] = 1

                memory_frame = MemoryFrame(new_state[i], old_state[i], new_time[i],
                                           old_time[i], key[i], reward[i],
                                           action_encoding, done[i], value[i])

                self.experience_memory.add_frame(memory_frame, i)

            self.experience_memory.increase_frame_pointer()
