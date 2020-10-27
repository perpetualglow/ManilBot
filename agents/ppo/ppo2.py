import torch
import torch.nn as nn
from torch.distributions import uniform
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

from agents.ppo.config import *

class PPO_Network2(nn.Module):

    def __init__(self, state_shape, action_num, mlp_layers):
        super(PPO_Network2, self).__init__()

        self.state_shape = state_shape
        self.action_num = action_num
        self.mlp_layers = mlp_layers

        layer_dims = self.mlp_layers
        self.ac = nn.Linear(layer_dims[0], layer_dims[1])
        self.ac_prelu = nn.PReLU()
        self.ac1 = nn.Linear(layer_dims[1], layer_dims[2])
        self.ac1_prelu = nn.PReLU()

        # Actor layers:
        self.a1 = nn.Linear(layer_dims[2] + action_num, action_num)

        # Critic layers:
        self.c1 = nn.Linear(layer_dims[2], layer_dims[2])
        self.c1_prelu = nn.PReLU()
        self.c2 = nn.Linear(layer_dims[2], 1)

    def forward(self, obs, legal_actions):
        ac = self.ac(obs)
        ac = self.ac_prelu(ac)
        ac = self.ac1(ac)
        ac = self.ac1_prelu(ac)

        if len(obs.shape) == 1:
            actor_out = torch.cat([ac, legal_actions], 0)
        else:
            actor_out = torch.cat([ac, legal_actions], 1)
        actor_out = self.a1(actor_out)
        actor_out = actor_out.softmax(dim=-1)

        critic = self.c1(ac)
        critic = self.c1_prelu(critic)
        critic = self.c2(critic)

        return actor_out, critic

class PPO2(nn.Module):

    def __init__(self, state_shape, action_num, mlp_layers):
        super(PPO2, self).__init__()

        self.state_shape = state_shape
        self.action_num = action_num
        self.mlp_layers = mlp_layers

        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers

        self.device = DEVICE
        self.actor_critic = PPO_Network2(state_shape, action_num, layer_dims).to(self.device)
        self.to(self.device)

        for p in self.actor_critic.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

    def eval_step(self, state, legal_actions):
        with torch.no_grad():
            obs = torch.from_numpy(np.array([state])).float().to(self.device)
            legals = torch.from_numpy(np.array([legal_actions])).float().detach().to(self.device)
            self.actor_critic.eval()
            action_probs, v = self.actor_critic(obs, legals)
            acts = action_probs * legals
            dist = Categorical(acts)
            action = dist.sample()
        self.actor_critic.train()
        return action.detach().cpu(), None, None, None, action_probs.detach().cpu().numpy()

    def forward(self, state, action=None, legal_actions=None):
        if action is None and legal_actions is None:
            obs = torch.from_numpy(np.squeeze(state)).float().detach().to(self.device)
            legals = np.zeros((obs.shape[0], self.action_num))
            legals = torch.from_numpy(np.squeeze(legals)).float().detach().to(self.device)
            a, v = self.actor_critic(obs, legals)
            return None, None, None, v.detach().cpu().numpy(), None
        if action is None and legal_actions is not None:
            obs = torch.from_numpy(np.squeeze(state)).float().detach().to(self.device)
            legals = torch.from_numpy(np.squeeze(legal_actions)).float().detach().to(self.device)
            action_probs, v = self.actor_critic(obs, legals)

            dist = Categorical(action_probs)
            acti = dist.sample()
            logprob = -dist.log_prob(acti)
            return acti.detach().cpu().numpy(), logprob.detach().cpu().numpy(), None, v.detach().cpu().numpy(), action_probs.detach().cpu().numpy()
        else:
            obs = torch.from_numpy(state).float().detach().to(self.device)
            legals = torch.from_numpy(legal_actions).float().detach().to(self.device)

            action_probs, v = self.actor_critic(obs, legals)
            dist = Categorical(action_probs)
            logprob = -dist.log_prob(action)
            entropy_loss = dist.entropy()
            return None, logprob, entropy_loss, v, action_probs