import torch

from tqdm import tqdm

from agent.utils import torch_device


class Trainer:
    def __init__(
        self,
        parallel_environment,
        experience_memory,
        agent_network,
        action_space,
        num_envs,
        experience_history_size,
        batch_size,
        num_of_epoches,
        total_timesteps,
    ):

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
        self.lr = 7e-4
        self.optim = torch.optim.Adam(
            self.agent_network.parameters(), lr=self.lr, eps=1e-5
        )

    def sample_action(self, actions):
        """
        Sample action from tensor of action probabilities or log probabilities.
        Input: torch.Tensor([8, 54])
        Return: torch.Tensor([8])
        """
        return self.distribution(actions).sample()

    def train(self):
        for timestep in range(0, self.total_timesteps, self.experience_history_size):
            self._fill_experience(timestep, len(self.action_space))
            self._update_observations(len(self.action_space))
            print(
                "Mean reward per episode: {:.2f}".format(
                    self.experience_memory.mean_reward()
                )
            )
            self.experience_memory.empty()
def _fill_experience(self, timestep, action_size):
        for step in tqdm(range(self.experience_history_size)):
            with torch.no_grad():
                if not timestep:
                    old_state, key, old_time = self.env.reset()
                    reward_action = torch.zeros((self.num_envs, action_size + 1)).to(
                        torch_device()
                    )
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action
                    )
                else:
                    last_rhs = self.experience_memory.last_hidden_state
                    old_state, reward_action, old_time = (
                        self.experience_memory.last_frames()
                    )
                    value, policy_acts, rhs = self.agent_network.act(
                        old_state, reward_action, last_rhs
                    )
                action = self.sample_action(policy_acts)
                new_actions = [self.action_space[act] for act in action]

                self.experience_memory.last_hidden_state = rhs
                new_state, key, new_time, reward, done = self.env.step(new_actions)

                action_encoding = torch.zeros((action_size, self.num_envs))
                policies = policy_acts.view(action_size, self.num_envs)
                for i in range(self.num_envs):
                    action_encoding[action[i], i] = 1

                self.experience_memory.add_experience(
                    new_state,
                    old_state,
                    new_time,
                    old_time,
                    key,
                    reward,
                    action_encoding,
                    done,
                    value,
                    policies,
                )
                self.experience_memory.increase_frame_pointer()
                timestep += 1

    def _update_observations(self, action_size):
        for _ in tqdm(range(self.num_of_epoches)):
            exp_batches = self.experience_memory.sample_observations(self.batch_size)
            states, reward_actions, action_indices, rewards, values, pixel_controls = (
                exp_batches
            )

            batch_advantages = []
            batch_returns = []

            for env in range(self.num_envs):
                returns = self.experience_memory.compute_returns(
                    rewards[env], values[env]
                )
                returns = torch.Tensor(returns).to(torch_device())
                adv = returns - values[env]

                batch_advantages.append(adv)
                batch_returns.append(returns)

            advantage = torch.cat(batch_advantages, dim=0).unsqueeze(-1)
            advantage = (advantage - torch.mean(advantage, dim=0)) / torch.std(
                advantage + 1e-6
            )

            new_value, policy_acts, _ = self.agent_network.act(states, reward_actions)
            q_aux = self.agent_network.pixel_control_act(states, reward_actions)

            returns = torch.cat(batch_returns, dim=0).to(torch_device())

            # Calculate q_aux and pc_returns
            a2c_loss = self.agent_network.a2c_loss(
                policy_acts, advantage, returns, new_value, action_indices
            )
            # pc_loss = self.agent_network.pc_loss(action_size, action_indices, q_aux, pc_returns)
            loss = a2c_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # ppo loss
