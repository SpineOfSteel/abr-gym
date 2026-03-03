import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---- constants ----
ACTION_EPS = 1e-6


# ---- classic Actor, Critic ----
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.s_info = int(state_dim[0])
        self.s_len = int(state_dim[1])
        self.action_dim = int(action_dim)
        self.input_channel = 1
        hidden = 128

        self.conv1 = nn.Conv1d(self.input_channel, hidden, 4)
        self.conv2 = nn.Conv1d(self.input_channel, hidden, 4)
        self.conv3 = nn.Conv1d(self.input_channel, hidden, 4)
        self.fc_scalar_1 = nn.Linear(self.input_channel, hidden)
        self.fc_scalar_2 = nn.Linear(self.input_channel, hidden)
        self.fc_scalar_3 = nn.Linear(self.input_channel, hidden)

        conv_hist = self.s_len - 4 + 1
        conv_sizes = self.action_dim - 4 + 1
        incoming_size = 2 * hidden * conv_hist + hidden * conv_sizes + 3 * hidden

        self.fc1 = nn.Linear(incoming_size, hidden)
        self.fc2 = nn.Linear(hidden, self.action_dim)

    def forward(self, inputs):
        x1 = F.relu(self.conv1(inputs[:, 2:3, :]))
        x2 = F.relu(self.conv2(inputs[:, 3:4, :]))
        x3 = F.relu(self.conv3(inputs[:, 4:5, :self.action_dim]))
        x4 = F.relu(self.fc_scalar_1(inputs[:, 0:1, -1]))
        x5 = F.relu(self.fc_scalar_2(inputs[:, 1:2, -1]))
        x6 = F.relu(self.fc_scalar_3(inputs[:, 5:6, -1]))

        x = torch.cat([
            torch.flatten(x1, 1),
            torch.flatten(x2, 1),
            torch.flatten(x3, 1),
            torch.flatten(x4, 1),
            torch.flatten(x5, 1),
            torch.flatten(x6, 1),
        ], dim=1)

        x = F.relu(self.fc1(x))
        probs = F.softmax(self.fc2(x), dim=1)
        probs = torch.clamp(probs, ACTION_EPS, 1.0 - ACTION_EPS)
        return probs / probs.sum(dim=1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.s_info = int(state_dim[0])
        self.s_len = int(state_dim[1])
        self.action_dim = int(action_dim)
        self.input_channel = 1
        hidden = 128

        self.conv1 = nn.Conv1d(self.input_channel, hidden, 4)
        self.conv2 = nn.Conv1d(self.input_channel, hidden, 4)
        self.conv3 = nn.Conv1d(self.input_channel, hidden, 4)
        self.fc_scalar_1 = nn.Linear(self.input_channel, hidden)
        self.fc_scalar_2 = nn.Linear(self.input_channel, hidden)
        self.fc_scalar_3 = nn.Linear(self.input_channel, hidden)

        conv_hist = self.s_len - 4 + 1
        conv_sizes = self.action_dim - 4 + 1
        incoming_size = 2 * hidden * conv_hist + hidden * conv_sizes + 3 * hidden

        self.fc1 = nn.Linear(incoming_size, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, inputs):
        x1 = F.relu(self.conv1(inputs[:, 2:3, :]))
        x2 = F.relu(self.conv2(inputs[:, 3:4, :]))
        x3 = F.relu(self.conv3(inputs[:, 4:5, :self.action_dim]))
        x4 = F.relu(self.fc_scalar_1(inputs[:, 0:1, -1]))
        x5 = F.relu(self.fc_scalar_2(inputs[:, 1:2, -1]))
        x6 = F.relu(self.fc_scalar_3(inputs[:, 5:6, -1]))

        x = torch.cat([
            torch.flatten(x1, 1),
            torch.flatten(x2, 1),
            torch.flatten(x3, 1),
            torch.flatten(x4, 1),
            torch.flatten(x5, 1),
            torch.flatten(x6, 1),
        ], dim=1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)