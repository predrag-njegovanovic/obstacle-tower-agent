import torch

from tqdm import tqdm

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
        for timestep in range(0, self.total_timesteps, self.experience_history_size):
            self._fill_experience(timestep, len(self.action_space))
            # update network
            self.experience_memory.empty()
            break

    def _fill_experience(self, timestep, action_size):
        for step in tqdm(range(self.experience_history_size)):
            with torch.no_grad():
                if not timestep:
                    old_state, key, old_time = self.env.reset()
                    reward_action = torch.zeros(
                        (self.num_envs, action_size + 1)).to(torch_device())
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action)
                else:
                    last_rhs = self.experience_memory.last_hidden_state
                    old_state, reward_action, old_time = self.experience_memory.last_frames()
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs)

                action = self.sample_action(policy_acts)
                new_actions = [self.action_space[act] for act in action]

                self.experience_memory.last_hidden_state = rhs
                new_state, key, new_time, reward, done = self.env.step(new_actions)
                action_encoding = torch.zeros((action_size + 1, self.num_envs))

                for i in range(self.num_envs):
                    action_encoding[action[i], i] = 1

                self.experience_memory.add_experience(new_state, old_state, new_time, old_time,
                                                      key, reward, action_encoding, done, value)
                self.experience_memory.increase_frame_pointer()
                timestep += 1
