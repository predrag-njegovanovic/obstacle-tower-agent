import torch

from agent import base_networks


class TowerAgent:
    def __init__(
        self,
        action_size,
        first_layer_filters,
        second_layer_filters,
        conv_output_size,
        hidden_state_size,
        entropy_coeff=0.01,
        value_coeff=0.5,
    ):

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

    def to_cuda(self):
        self.conv_network.cuda()
        self.lstm_network.cuda()
        self.pc_network.cuda()
        self.value.cuda()
        self.policy.cuda()

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

        q_aux = self.pc_network(features)
        return q_aux

    def a2c_loss(self, policy_logs, advantage, returns, values):
        policy_loss = self._policy_loss(policy_logs, advantage)
        value_loss = self._value_loss(returns, values)
        entropy = self._entropy(policy_logs)

        loss = (
            policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy
        )
        return loss

    def pc_loss(self):
        pass

    def _policy_loss(self, policy_logs, adventage):
        return -torch.mean(adventage * policy_logs)

    def _value_loss(self, returns, values):
        return torch.mean(torch.pow(returns - values, 2))

    def _entropy(self, policy_logs):
        policy = torch.log(torch.clamp(policy_logs, 1e-20, 1.0))

        return -torch.mean(policy_logs * policy)
