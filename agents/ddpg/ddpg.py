import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

class Actor(nn.Module):

    def __init__(self, state_shape, action_num, mlp_layers, device):
        super(Actor, self).__init__()

        self.state_shape = state_shape
        self.action_num = action_num
        self.mlp_layers = mlp_layers

        layer_dims = self.mlp_layers
        self.linear1 = nn.Linear(layer_dims[0], layer_dims[1])
        self.linear2 = nn.Linear(layer_dims[1], layer_dims[2])
        self.linear3 = nn.Linear(layer_dims[2], action_num)
        self.device = device

    def forward(self, obs, legal_actions, mean=False):
        # legal_actions_t = torch.from_numpy(legal_actions).float().detach().to(self.device)
        # obs = torch.from_numpy(obs).float().detach().to(self.device)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        if not mean:
            res = x + legal_actions
            return res
        else:
            return x

class Critic(nn.Module):

    def __init__(self, state_shape, action_num, mlp_layers, device):
        super(Critic, self).__init__()

        self.state_shape = state_shape
        self.action_num = action_num
        self.mlp_layers = mlp_layers

        layer_dims = self.mlp_layers
        self.linear1 = nn.Linear(layer_dims[0], layer_dims[1])
        self.linear2 = nn.Linear(layer_dims[1] + 4 * action_num, layer_dims[2])
        self.linear3 = nn.Linear(layer_dims[2], layer_dims[3])
        self.linear4 = nn.Linear(layer_dims[3], layer_dims[4])
        self.linear5 = nn.Linear(layer_dims[4], 1)

        self.device = device

    def forward(self, x, actions):
        x = F.relu(self.linear1(x))
        xa = torch.cat((x,actions), dim=-1)
        xa = F.relu(self.linear2(xa))
        xa = F.relu(self.linear3(xa))
        xa = F.relu(self.linear4(xa))
        qval = self.linear5(xa)
        return qval
