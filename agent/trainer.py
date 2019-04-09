from tower_agent import TowerAgent


class Trainer:
    def __init__(self,
                 parallel_environment,
                 experience_memory,
                 agent_network,
                 experience_history_size,
                 batch_size,
                 num_of_epoches,
                 total_timesteps):

        self.parallel_environment = parallel_environment
        self.experience_memory = experience_memory
        self.agent_network = agent_network
        self.experience_history_size = experience_history_size
        self.batch_size = batch_size
        self.num_of_epoches = num_of_epoches
        self.total_timesteps = total_timesteps

    def train(self):
        for _ in range(0, self.total_timesteps, self.experience_history_size):
            # fill experience
            # update network
            pass

    def _fill_experience(self):
        state, key, time = self.parallel_environment.reset()
        for step in range(self.experience_history_size):
            if step == 0:
                # use state from reset now
                # time diff = 0
            else:
                # use last state from memory
