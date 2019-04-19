import torch

from agent import base_networks


class TowerAgent(torch.nn.Module):
    def __init__(
        self,
        action_size,
        first_layer_filters,
        second_layer_filters,
        conv_output_size,
        hidden_state_size,
        entropy_coeff=0.01,
        value_coeff=0.5,
        pc_lambda=0.99,
        ppo_epsilon=0.2,
    ):

        super(TowerAgent, self).__init__()

        self.conv_network = base_networks.ConvNetwork(
            first_layer_filters, second_layer_filters, conv_output_size
        )
        self.lstm_network = base_networks.LSTMNetwork(
            conv_output_size, hidden_state_size, action_size
        )
        self.pc_network = base_networks.PixelControlNetwork(action_size)

        self.value = base_networks.ValueNetwork()
        self.policy = base_networks.PolicyNetwork(action_size)
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.pc_lambda = pc_lambda
        self.ppo_epsilon = ppo_epsilon

    def to_cuda(self):
        self.conv_network.cuda()
        self.lstm_network.cuda()
        self.pc_network.cuda()
        self.value.cuda()
        self.policy.cuda()

    # def parameters(self):
    #     return (
    #         list(self.conv_network.parameters())
    #         + list(self.lstm_network.parameters())
    #         + list(self.value.parameters())
    #         + list(self.policy.parameters())
    #         + list(self.pc_network.parameters())
    #     )

    def act(self, state, reward_and_last_action, last_hidden_state=None):
        """
        Run batch of states (3-channel images) through network to get
        estimated value and policy logs.
        """
        conv_features = self.conv_network(state)
        features, hidden_state = self.lstm_network(
            conv_features, reward_and_last_action, last_hidden_state
        )

        value = self.value(features)
        policy = self.policy(features)

        return value, policy, hidden_state

    def pixel_control_act(self, state, reward_and_last_action, last_hidden_state=None):
        """
        Run batch of states (sampled from experience memory) through network to get
        auxiliary Q value.
        """
        conv_features = self.conv_network(state)
        features, hidden_state = self.lstm_network(
            conv_features, reward_and_last_action, last_hidden_state
        )

        q_aux, q_aux_max = self.pc_network(features)
        return q_aux, q_aux_max

    def ppo_loss(self, old_policy, policy, advantage, returns, values, action_indices):
        policy_loss = self.ppo_policy_loss(
            old_policy, policy, advantage, action_indices
        )
        value_loss = self.value_loss(returns, values)
        entropy = self.entropy(policy)

        loss = (
            policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        )

        return loss, policy_loss, value_loss, entropy

    def a2c_loss(self, policy_logs, advantage, returns, values, action_indices):
        policy_loss = self.policy_loss(policy_logs, advantage, action_indices)
        value_loss = self.value_loss(returns, values)
        entropy = self.entropy(policy_logs)

        loss = (
            policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        )
        return loss, policy_loss, value_loss, entropy

    def pc_loss(self, action_size, action_indices, q_aux, pc_returns):
        reshaped_indices = action_indices.view(-1, action_size, 1, 1)
        pc_q_aux = torch.mul(q_aux, reshaped_indices)

        pc_q_aux_sum = torch.sum(pc_q_aux, dim=1, keepdim=False)

        # try with torch.sum instead of torch.mean
        pc_loss = self.pc_lambda * torch.mean(torch.pow(pc_returns * pc_q_aux_sum, 2))
        return pc_loss

    def v_loss(self, v_returns, new_value):
        v_loss = torch.mean(torch.pow(v_returns - new_value, 2))
        return v_loss

    def policy_loss(self, policy_logs, adventage, action_indices):
        pi_logs = torch.sum(torch.mul(policy_logs, action_indices), 1)
        return -torch.mean(adventage * pi_logs)

    def ppo_policy_loss(self, old_policy, policy, advantage, action_indices):
        pi_logs = torch.sum(torch.mul(policy, action_indices), 1)
        old_pi_logs = torch.sum(torch.mul(old_policy, action_indices), 1)

        ratio = torch.exp(pi_logs - old_pi_logs)
        ratio_term = ratio * advantage
        clamp = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
        clamp_term = clamp * advantage

        policy_loss = -torch.min(ratio_term, clamp_term).mean()
        return policy_loss

    def value_loss(self, returns, values):
        return torch.mean(torch.pow(returns - values, 2))

    def entropy(self, policy_logs):
        policy = torch.exp(policy_logs)

        # try with torch.sum instead of torch.mean
        return torch.mean(policy_logs * policy)
