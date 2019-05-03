import torch


class ConvNetwork(torch.nn.Module):
    def __init__(self, first_layer_filters, second_layer_filters, out_features):
        super(ConvNetwork, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=first_layer_filters,
            kernel_size=(8, 8),
            stride=4,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=first_layer_filters,
            out_channels=second_layer_filters,
            kernel_size=(4, 4),
            stride=2,
        )

        self.fully_connected = torch.nn.Linear(2592, out_features)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, inputs):
        new_input = inputs.type(torch.float32)
        new_input = new_input / 255

        conv1_out = self.conv1(new_input)
        self.relu(conv1_out)

        conv2_out = self.conv2(conv1_out)
        self.relu(conv2_out)

        fc_input = conv2_out.view(conv2_out.size(0), -1)
        linear_out = self.fully_connected(fc_input)
        self.relu(linear_out)

        return linear_out


class LSTMNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_state_size, action_size):
        super(LSTMNetwork, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_state_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, inputs, reward_and_last_action, last_hidden_state):
        features = torch.cat((inputs, reward_and_last_action), dim=1)
        # 8 x 311
        batch_seq = inputs.unsqueeze(1)
        # 8 x 1 x 311 (batch, seq, in)
        output, hidden_state = self.lstm(batch_seq, last_hidden_state)
        return output.squeeze(1), hidden_state


class ValueNetwork(torch.nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.value = torch.nn.Linear(in_features=256, out_features=1)

    def forward(self, inputs):
        """
        Return estimated value V(s).
        """
        # create Tensor([8]) out of Tensor(8 x 1)
        value = torch.squeeze(self.value(inputs))
        return value


class PolicyNetwork(torch.nn.Module):
    def __init__(self, action_size):
        super(PolicyNetwork, self).__init__()

        self.fully_connected = torch.nn.Linear(
            in_features=256, out_features=action_size
        )

        self.policy = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        """
        Return action probabilities.
        """
        fc_out = self.fully_connected(inputs)
        policy = self.policy(fc_out)
        return policy


class FeatureExtractor(torch.nn.Module):
    def __init__(self, num_of_filters, output_size):
        super(FeatureExtractor, self).__init__()

        self.conv_f = torch.nn.Conv2d(
            3, num_of_filters, kernel_size=3, stride=2, padding=1)
        self.conv_s = torch.nn.Conv2d(
            num_of_filters, num_of_filters, kernel_size=3, stride=2, padding=1)
        self.conv_t = torch.nn.Conv2d(
            num_of_filters, num_of_filters, kernel_size=3, stride=2, padding=1)
        self.conv_final = torch.nn.Conv2d(
            num_of_filters, num_of_filters, kernel_size=3, stride=2, padding=1)

        self.linear = torch.nn.Linear(32 * 6 * 6, output_size)

        self.elu = torch.nn.ELU(inplace=True)

    def forward(self, state):
        state = state.type(torch.float32)
        state = state / 255

        f_output = self.conv_f(state)
        self.elu(f_output)
        s_output = self.conv_s(f_output)
        self.elu(s_output)
        t_output = self.conv_t(s_output)
        self.elu(t_output)
        conv_final = self.conv_final(t_output)
        self.elu(conv_final)

        flatten = conv_final.view(conv_final.size(0), -1)
        features = self.linear(flatten)
        self.elu(features)

        return features


class ForwardModel(torch.nn.Module):
    def __init__(self, f_layer_size):
        super(ForwardModel, self).__init__()

        self.f_layer = torch.nn.Linear(f_layer_size, 256)
        self.s_layer = torch.nn.Linear(256, 288)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, features, action_indices):
        concat_features = torch.cat(
            (features, action_indices), dim=1)

        self.relu(concat_features)
        intermediate_res = self.f_layer(concat_features)
        self.relu(intermediate_res)
        pred_state = self.s_layer(intermediate_res)

        return pred_state


class InverseModel(torch.nn.Module):
    def __init__(self, f_layer_size, action_size):
        super(InverseModel, self).__init__()

        self.f_layer = torch.nn.Linear(f_layer_size, 256)
        self.s_layer = torch.nn.Linear(256, action_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, state_features, new_state_features):
        concat_features = torch.cat((state_features, new_state_features), dim=1)

        f_output = self.f_layer(concat_features)
        s_output = self.s_layer(f_output)
        pred_actions = self.softmax(s_output)

        return pred_actions


class PixelControlNetwork(torch.nn.Module):
    def __init__(self, action_size):
        super(PixelControlNetwork, self).__init__()

        self.fully_connected = torch.nn.Linear(in_features=256, out_features=9 * 9 * 32)
        self.deconv_value = torch.nn.ConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=(4, 4), stride=2
        )
        self.deconv_adv = torch.nn.ConvTranspose2d(
            in_channels=32, out_channels=action_size, kernel_size=(4, 4), stride=2
        )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, inputs):
        linear_out = self.fully_connected(inputs)
        linear_out = linear_out.view([-1, 32, 9, 9])

        self.relu(linear_out)

        value = self.deconv_value(linear_out)
        advantage = self.deconv_adv(linear_out)

        advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
        q_aux = value + advantage - advantage_mean
        q_aux_max = torch.max(q_aux, dim=1, keepdim=False)[0]
        return q_aux, q_aux_max
