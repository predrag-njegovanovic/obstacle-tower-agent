import base_networks


class TowerAgent:
    def __init__(self,
                 action_size,
                 first_layer_filters,
                 second_layer_filters,
                 conv_output_size,
                 lstm_input_size,
                 hidden_state_size,
                 value_policy_input_size,
                 pixel_control_input):

        assert conv_output_size == lstm_input_size, "Convolution output must be the same as LSTM input"

        self.conv_network = base_networks.ConvNetwork(
            first_layer_filters, second_layer_filters, conv_output_size)
        self.lstm_network = base_networks.LSTMNetwork(
            lstm_input_size, hidden_state_size, action_size)
        self.pc_network = base_networks.PixelControlNetwork(
            pixel_control_input, action_size)

        self.value = base_networks.ValueNetwork(value_policy_input_size)
        self.policy = base_networks.PolicyNetwork(value_policy_input_size, action_size)

    def act(self, state, reward_and_last_action, last_hidden_state=None):
        """
        Run batch of states (3-channel images) through network to get
        estimated value and policy logs.
        """
        conv_features = self.conv_network(state)
        features, hidden_state = self.lstm_network(
            conv_features, reward_and_last_action, last_hidden_state)

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
            conv_features, reward_and_last_action, last_hidden_state)

        q_aux = self.pc_network(features)
        return q_aux
