import torch


def _init_module_weights(module, gain="relu"):
    gain_init = 1 if gain == "constant" else torch.nn.init.calculate_gain(gain)
    torch.nn.init.orthogonal_(module.weight.data, gain=gain_init)
    torch.nn.init.constant_(module.bias.data, 0)
    return module


def _init_gru(gru_module):
    for name, param in gru_module.named_parameters():
        if "bias" in name:
            torch.nn.init.constant_(param, 0)
        elif "weight" in name:
            torch.nn.init.orthogonal_(param)
    return gru_module


class BaseNetwork(torch.nn.Module):
    def __init__(
        self, first_layer_filters, second_layer_filters, out_features, obs_mean, obs_std
    ):
        super(BaseNetwork, self).__init__()

        self.conv1 = _init_module_weights(
            torch.nn.Conv2d(
                in_channels=3, out_channels=first_layer_filters, kernel_size=8, stride=4
            ),
            gain="leaky_relu",
        )
        self.conv2 = _init_module_weights(
            torch.nn.Conv2d(
                in_channels=first_layer_filters,
                out_channels=second_layer_filters,
                kernel_size=4,
                stride=2,
            ),
            gain="leaky_relu",
        )
        self.conv3 = _init_module_weights(
            torch.nn.Conv2d(
                in_channels=second_layer_filters,
                out_channels=first_layer_filters,
                kernel_size=3,
                stride=1,
            ),
            gain="leaky_relu",
        )

        self.mean = obs_mean
        self.std = obs_std

        self.fully_connected = _init_module_weights(
            torch.nn.Linear(32 * 7 * 7, out_features), gain="leaky_relu"
        )
        self.lrelu = torch.nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        new_input = inputs.type(torch.float32)
        new_input = (new_input - self.mean) / (self.std + 1e-6)
        # new_input = new_input / 255

        conv1_out = self.conv1(new_input)
        self.lrelu(conv1_out)

        conv2_out = self.conv2(conv1_out)
        self.lrelu(conv2_out)

        conv3_out = self.conv3(conv2_out)
        self.lrelu(conv3_out)

        fc_input = conv3_out.view(conv3_out.size(0), -1)

        linear_out = self.fully_connected(fc_input)
        self.lrelu(linear_out)

        return linear_out


class GRUNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_state_size, action_size):
        super(GRUNetwork, self).__init__()

        self.gru = _init_gru(
            torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_state_size,
                num_layers=1,
                batch_first=True,
            )
        )

    def forward(self, inputs, reward_and_last_action, last_hidden_state):
        features = torch.cat((inputs, reward_and_last_action), dim=1)
        # 8 x 311

        if inputs.size(0) > 8:
            batch_seq = inputs.unsqueeze(0)
        else:
            batch_seq = inputs.unsqueeze(1)
        # 8 x 1 x 311 (batch, seq, in)
        output, hidden_state = self.gru(batch_seq, last_hidden_state)

        if inputs.size(0) > 8:
            output = output.squeeze(0)
        else:
            output = output.squeeze(1)

        return output, hidden_state


class ValueNetwork(torch.nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.value = _init_module_weights(
            torch.nn.Linear(in_features=512, out_features=1), gain="constant"
        )

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

        self.fully_connected = _init_module_weights(
            torch.nn.Linear(in_features=512, out_features=action_size), gain="constant"
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
    def __init__(self, num_of_filters, output_size, obs_mean, obs_std):
        super(FeatureExtractor, self).__init__()

        self.conv_f = torch.nn.Conv2d(
            3, num_of_filters, kernel_size=3, stride=2, padding=1
        )
        self.conv_s = torch.nn.Conv2d(
            num_of_filters, num_of_filters, kernel_size=3, stride=2, padding=1
        )
        self.conv_t = torch.nn.Conv2d(
            num_of_filters, num_of_filters * 2, kernel_size=3, stride=2, padding=1
        )
        self.conv_final = torch.nn.Conv2d(
            num_of_filters * 2, num_of_filters * 2, kernel_size=3, stride=2, padding=1
        )
        self.linear = torch.nn.Linear(in_features=64 * 6 * 6, out_features=288)

        self.lrelu = torch.nn.LeakyReLU(inplace=True)

        self.bn1 = torch.nn.BatchNorm2d(num_of_filters)
        self.bn2 = torch.nn.BatchNorm2d(num_of_filters)
        self.bn3 = torch.nn.BatchNorm2d(num_of_filters * 2)
        self.bn4 = torch.nn.BatchNorm2d(num_of_filters * 2)
        self.mean = obs_mean
        self.std = obs_std

        torch.nn.init.xavier_uniform_(self.conv_f.weight)
        torch.nn.init.xavier_uniform_(self.conv_s.weight)
        torch.nn.init.xavier_uniform_(self.conv_t.weight)
        torch.nn.init.xavier_uniform_(self.conv_final.weight)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, state):
        state = state.type(torch.float32)
        state = (state - self.mean) / (self.std + 1e-6)

        f_output = self.conv_f(state)
        f_output = self.bn1(f_output)
        self.lrelu(f_output)
        s_output = self.conv_s(f_output)
        s_output = self.bn2(s_output)
        self.lrelu(s_output)
        t_output = self.conv_t(s_output)
        t_output = self.bn3(t_output)
        self.lrelu(t_output)
        conv_final = self.conv_final(t_output)
        conv_final = self.bn4(conv_final)
        self.lrelu(conv_final)

        flatten = conv_final.view(conv_final.size(0), -1)
        features = self.linear(flatten)
        self.lrelu(features)

        return features


class ForwardModel(torch.nn.Module):
    def __init__(self, f_layer_size):
        super(ForwardModel, self).__init__()

        self.f_layer = torch.nn.Linear(f_layer_size, 256)
        self.hidden = torch.nn.Linear(256, 256 * 2)
        self.s_layer = torch.nn.Linear(256 * 2, 288)
        self.lrelu = torch.nn.LeakyReLU(inplace=True)

        torch.nn.init.xavier_uniform_(self.f_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.s_layer.weight)

    def forward(self, features, action_indices):
        concat_features = torch.cat((features, action_indices), dim=1)

        intermediate_res = self.f_layer(concat_features)
        self.lrelu(intermediate_res)
        hidden_f = self.hidden(intermediate_res)
        self.lrelu(hidden_f)
        pred_state = self.s_layer(hidden_f)

        return pred_state


class InverseModel(torch.nn.Module):
    def __init__(self, f_layer_size, action_size):
        super(InverseModel, self).__init__()

        self.f_layer = torch.nn.Linear(f_layer_size, 256)
        self.hidden_1 = torch.nn.Linear(256, 256 * 2)
        self.hidden_2 = torch.nn.Linear(256 * 2, 256 * 2)
        self.s_layer = torch.nn.Linear(256 * 2, action_size)

        self.lrelu = torch.nn.LeakyReLU(inplace=True)
        self.softmax = torch.nn.Softmax(dim=1)

        torch.nn.init.xavier_uniform_(self.f_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden_1.weight)
        torch.nn.init.xavier_uniform_(self.hidden_2.weight)
        torch.nn.init.xavier_uniform_(self.s_layer.weight)

    def forward(self, state_features, new_state_features):
        concat_features = torch.cat((state_features, new_state_features), dim=1)

        f_output = self.f_layer(concat_features)
        self.lrelu(f_output)
        hidden_1_out = self.hidden_1(f_output)
        self.lrelu(hidden_1_out)
        hidden_2_out = self.hidden_2(hidden_1_out)
        self.lrelu(hidden_2_out)
        s_output = self.s_layer(hidden_2_out)
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
