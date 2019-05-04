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
        feature_ext_filters,
        feature_output_size,
        forward_model_f_layer,
        inverse_model_f_layer,
        obs_mean,
        obs_std,
        entropy_coeff=0.01,
        value_coeff=0.5,
        pc_lambda=0.01,
        ppo_epsilon=0.2,
        beta=0.2,
        isc_lambda=0.1
    ):

        super(TowerAgent, self).__init__()

        self.conv_network = base_networks.BaseNetwork(
            first_layer_filters, second_layer_filters, conv_output_size, obs_mean, obs_std
        )
        self.lstm_network = base_networks.LSTMNetwork(
            conv_output_size, hidden_state_size, action_size
        )
        self.feature_extractor = base_networks.FeatureExtractor(
            feature_ext_filters, feature_output_size, obs_mean, obs_std)
        self.forward_model = base_networks.ForwardModel(forward_model_f_layer)
        self.inverse_model = base_networks.InverseModel(
            inverse_model_f_layer, action_size)
        # self.pc_network = base_networks.PixelControlNetwork(action_size)

        self.value = base_networks.ValueNetwork()
        self.policy = base_networks.PolicyNetwork(action_size)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.ent_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.pc_lambda = pc_lambda
        self.ppo_epsilon = ppo_epsilon
        self.beta = beta
        self.isc_lambda = isc_lambda

    def to_cuda(self):
        self.conv_network.cuda()
        self.lstm_network.cuda()
        # self.pc_network.cuda()
        self.value.cuda()
        self.policy.cuda()
        self.feature_extractor.cuda()
        self.forward_model.cuda()
        self.inverse_model.cuda()

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

    def icm_act(self, state, new_state, action_indices, eta=0.01):
        state_features = self.feature_extractor(state)
        new_state_features = self.feature_extractor(new_state)

        pred_state = self.forward_model(state_features, action_indices)

        intrinsic_reward = (eta / 2) * \
            torch.mean(torch.pow(pred_state - new_state_features, 2), dim=1)

        return intrinsic_reward, state_features, new_state_features

    def forward_act(self, batch_state_features, batch_action_indices):
        batch_pred_state = self.forward_model(batch_state_features, batch_action_indices)
        return batch_pred_state

    def inverse_act(self, batch_state_features, batch_new_state_features):
        batch_pred_acts = self.inverse_model(
            batch_state_features, batch_new_state_features)
        return batch_pred_acts

    def ppo_loss(self, old_policy, policy, advantage, returns, values, action_indices):
        policy_loss = self.ppo_policy_loss(
            old_policy, policy, advantage, action_indices
        )
        value_loss = self.value_loss(returns, values)
        entropy = self.entropy(policy)

        loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy

        return loss, policy_loss, value_loss, entropy

    def a2c_loss(self, policy, advantage, returns, values,
                 action_indices, new_state_features, new_state_preds, pred_acts):

        policy_loss = self.policy_loss(policy, advantage, action_indices)
        value_loss = self.value_loss(returns, values)
        entropy = self.entropy(policy)

        a2c_loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy
        fwd_loss = self.forward_loss(new_state_features, new_state_preds)
        inv_loss = self.inverse_loss(pred_acts, action_indices.detach())

        loss = self.isc_lambda * a2c_loss + \
            (1 - self.beta) * inv_loss + self.beta * fwd_loss

        return loss, policy_loss, value_loss, entropy, fwd_loss, inv_loss

    def pc_loss(self, action_size, action_indices, q_aux, pc_returns):
        reshaped_indices = action_indices.view(-1, action_size, 1, 1).cuda()
        pc_q_aux = torch.mul(q_aux, reshaped_indices)

        pc_q_aux_sum = torch.sum(pc_q_aux, dim=1, keepdim=False)

        pc_loss = 0.5 * \
            self.pc_lambda * torch.mean(torch.pow(pc_returns - pc_q_aux_sum, 2))
        return pc_loss

    def forward_loss(self, new_state_features, new_state_pred):
        fwd_loss = 0.5 * torch.mean(torch.pow(new_state_features - new_state_pred, 2))
        return fwd_loss

    def inverse_loss(self, pred_acts, action_indices):
        inv_loss = self.cross_entropy_loss(pred_acts, torch.argmax(action_indices, dim=1))

        return inv_loss

    def v_loss(self, v_returns, new_value):
        v_loss = 0.5 * torch.mean(torch.pow(v_returns - new_value, 2))
        return v_loss

    def policy_loss(self, policy, adventage, action_indices):
        policy_logs = torch.log(torch.clamp(policy, 1e-20, 1.0))

        pi_logs = torch.sum(torch.mul(policy_logs, action_indices.cuda()), 1)
        policy_loss = -torch.mean(adventage * pi_logs)
        return policy_loss

    def ppo_policy_loss(self, old_policy, policy, advantage, action_indices):
        policy = torch.log(torch.clamp(policy, 1e-20, 1.0))
        old_policy = torch.log(torch.clamp(old_policy, 1e-20, 1.0))

        pi_logs = torch.sum(torch.mul(policy, action_indices), 1)
        old_pi_logs = torch.sum(torch.mul(old_policy, action_indices), 1)

        ratio = torch.exp(pi_logs - old_pi_logs)
        ratio_term = ratio * advantage
        clamp = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)
        clamp_term = clamp * advantage

        policy_loss = -torch.min(ratio_term, clamp_term).mean()
        return policy_loss

    def value_loss(self, returns, values):
        return 0.5 * torch.mean(torch.pow(returns - values, 2))

    def entropy(self, policy):
        dist = torch.distributions.Categorical
        return dist(probs=policy).entropy().mean()
