import torch


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
        for _ in range(0, self.total_timesteps, self.experience_history_size):
            self._fill_experience()
            # update network
            pass

    def _fill_experience(self):
        action_size = len(self.action_space)
        old_state, key, time = self.env.reset()
        for step in range(self.experience_history_size):
            if step == 0:
                reward_action = torch.zeros((self.num_envs, action_size + 1))
                # (1x8) (8x54) tuple((,))
                value, policy_acts, rhs = self.agent_network.act(old_state, reward_action)
                import pdb
                pdb.set_trace()
                # (1x8)
                action = self.sample_action(policy_acts)
            else:
                pass
                # load last hidden state from memory
                # load last state from memory
                # load last reward_action from memory

            new_state, key, time, reward, done = self.env(action)
            break
            # create memory frame now (new_state, reward + action, old_state, time_diff)
