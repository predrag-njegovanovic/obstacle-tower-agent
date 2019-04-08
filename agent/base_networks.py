import torch


class ConvNetwork(torch.nn.Module):
    def __init__(self, first_layer_filters, second_layer_filters, out_features):
        self.conv1 = torch.nn.Conv2d(in_channels=3,
                                     out_channels=first_layer_filters,
                                     kernel_size=(8, 8),
                                     stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=first_layer_filters,
                                     out_channels=second_layer_filters,
                                     kernel_size=(4, 4),
                                     stride=2)

        self.fully_connected = torch.nn.Linear(2592, out_features)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, inputs):
        conv1_out = self.conv1(inputs)
        self.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        self.relu(conv2_out)
        linear_out = self.fully_connected(conv2_out)
        self.relu(linear_out)
        return linear_out


class LSTMNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_state_size, action_size):
        self.lstm = torch.nn.LSTM(input_size=input_size + action_size + 1,
                                  hidden_size=hidden_state_size,
                                  batch_first=True)

    def forward(self, inputs, reward_and_last_action, last_hidden_state=None):
        features = torch.cat([inputs, reward_and_last_action, 1], 1)
        # 8 x 1 x 256
        output, hidden_state = self.lstm(features, last_hidden_state)
        return output, hidden_state


class ValueNetwork(torch.nn.Module):
    def __init__(self, input_size):
        self.value = torch.nn.Linear(in_features=input_size, out_features=1)

    def forward(self, inputs):
        """
        Return estimated value V(s).
        """
        return self.value(inputs)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, action_size):
        self.fully_connected = torch.nn.Linear(in_features=input_size,
                                               out_features=action_size)
        self.policy = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        """
        Return Log(pi).
        """
        return self.policy(self.fully_connected(inputs))


class PixelControlNetwork(torch.nn.Module):
    def __init__(self, input_size, action_size):
        self.fully_connected = torch.nn.Linear(in_features=input_size,
                                               out_features=7 * 7 * 32)
        self.deconv_value = torch.nn.ConvTranspose2d(in_channels=32,
                                                     out_channels=1,
                                                     kernel_size=(4, 4),
                                                     stride=2)
        self.deconv_adv = torch.nn.ConvTranspose2d(in_channels=32,
                                                   out_channels=action_size,
                                                   kernel_size=(4, 4),
                                                   stride=2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, inputs):
        linear_out = self.fully_connected(inputs)
        linear_out = linear_out.view([-1, 32, 7, 7])

        self.relu(linear_out)

        value = self.deconv_value(linear_out)
        advantage = self.deconv_adv(linear_out)

        advantage_mean = torch.mean(advantage, dim=1, keepdim=True)
        q_aux = value + advantage - advantage_mean
        # q_aux_max = torch.max(q_aux, dim=1, keepdim=False)[0]
        return q_aux
