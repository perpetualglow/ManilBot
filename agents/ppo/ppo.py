import torch
import torch.nn as nn
from torch.distributions import uniform
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agents.ppo.config import *
import config as manil_config


class PPO_Network(nn.Module):

    def __init__(self, state_shape, action_num, mlp_layers, first_layer, actor):
        super(PPO_Network, self).__init__()

        self.state_shape = state_shape
        self.action_num = action_num
        self.mlp_layers = mlp_layers

        layer_dims = self.mlp_layers
        fc = [nn.Flatten()]
        if actor == True:
            for i in range(0, len(layer_dims) - 1):
                fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                fc.append(nn.PReLU())
            fc.append(nn.Linear(layer_dims[-1], action_num))
            fc.append(nn.ReLU())
            self.fc_layers = nn.Sequential(*fc)
        else:
            for i in range(0, len(layer_dims) - 1):
                fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                fc.append(nn.ReLU())
            fc.append(nn.Linear(layer_dims[-1], action_num))
            self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        return self.fc_layers(s)

class Multi_Critic(nn.Module):
    def __init__(self, state_shape, perfect_state_shape, mlp_layers):
        super(Multi_Critic, self).__init__()

        self.state_shape = state_shape
        self.perfect_state_shape = perfect_state_shape
        self.mlp_layers = mlp_layers

        layer_dims = self.mlp_layers
        fc = [nn.Flatten()]
        for i in range(0, len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            fc.append(nn.PReLU())
        fc.append(nn.Linear(layer_dims[-1], 1))
        fc.append(nn.PReLU())
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, s):
        return self.fc_layers(s)



"""### PPO Model"""


class PPO(nn.Module):

    def __init__(self, state_shape, perfect_state_shape, action_num, mlp_layers):
        super(PPO, self).__init__()
        self.state_shape = state_shape
        self.perfect_state_shape = perfect_state_shape
        self.action_num = action_num
        self.mlp_layers = mlp_layers
        self.device = DEVICE
        self.lower = 0.01
        self.higher = 0.99

        layer_dims_actor = [np.prod(self.state_shape)] + self.mlp_layers
        layer_dims_critic = [np.prod(self.state_shape)] + self.mlp_layers
        self.actor = PPO_Network(state_shape, action_num, layer_dims_actor, None, True).to(self.device)
        self.critic = PPO_Network(state_shape, 1, layer_dims_critic, None, False).to(self.device)
        self.to(self.device)

        for p in self.actor.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
                #nn.init.zeros_(p.data)
        for p in self.critic.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)
                #nn.init.zeros_(p.data)

    def value(self, states):
        obs = torch.from_numpy(states).float().detach().to(self.device)
        v = self.critic(obs)
        return v.detach().cpu().numpy()

    def value_ppo(self, states):
        obs = torch.from_numpy(states).float().detach().to(self.device)
        v = self.critic(obs)
        return v

    def eval_step(self, state, legal_actions):
        with torch.no_grad():
            obs = torch.from_numpy(np.array(state)).float().to(self.device)
            self.actor.eval()
            a = self.actor(obs)
            legal_actions_t = torch.from_numpy(legal_actions).float().detach().to(self.device)
            t = torch.add(a, legal_actions_t)
            action_probs = F.softmax(t, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
        self.actor.train()
        return action.detach().cpu(), None, None, None, action_probs.detach().cpu().numpy()

    def forward(self, state, action=None, legal_actions=None):
        if action is None and legal_actions is None:
            obs = torch.from_numpy(state).float().detach().to(self.device)
            v = self.critic(obs)
            return None, None, None, v.detach().cpu().numpy(), None
        if action is None and legal_actions is not None:
            obs = torch.from_numpy(state).float().detach().to(self.device)
            legal_actions_t = torch.from_numpy(legal_actions).float().detach().to(self.device)
            a = self.actor(obs)
            v = self.critic(obs)
            t = torch.add(a, legal_actions_t)
            action_probs = F.softmax(t, dim=-1)
            dist = Categorical(action_probs)
            acti = dist.sample()
            logprob = dist.log_prob(acti)
            '''
            dist2 = uniform.Uniform((torch.ones(legal_actions.shape) * self.lower),
                                                        torch.ones(legal_actions.shape) * self.higher)
            act = torch.argmax(t - torch.log(-torch.log(dist2.sample().to(self.device))), dim=-1)
            el = torch.exp(t - torch.max(t, dim=-1, keepdim=True)[0])
            z = torch.sum(el, dim=-1, keepdim=True)
            p0 = el / z
            onehot2 = torch.nn.functional.one_hot(act, self.action_num).to(self.device)
            neglogpac = -torch.log(torch.sum(p0 * onehot2, dim=-1) + 1e-9)
            '''
            #return act.detach().cpu().numpy(), neglogpac.detach().cpu().numpy(), None, v.detach().cpu().numpy(), p0.detach().cpu().numpy()
            return acti.detach().cpu().numpy(), logprob.detach().cpu().numpy(), None, v.detach().cpu().numpy(), action_probs.detach().cpu().numpy()
            # return acti.detach().cpu().numpy(), logprob.detach().cpu().numpy(), None, None, action_probs.detach().cpu().numpy()
        else:
            obs = torch.from_numpy(state).float().detach().to(self.device)
            legal_actions_t = torch.from_numpy(legal_actions).float().detach().to(self.device)

            a = self.actor(obs)
            v = self.critic(obs)
            t = torch.add(a, legal_actions_t)

            action_probs = F.softmax(t, dim=-1)
            dist = Categorical(action_probs)
            neglogpac = dist.log_prob(action)
            entropy_loss = dist.entropy()

            '''
            el = torch.exp(t - torch.max(t, dim=-1, keepdim=True)[0])
            z0 = torch.sum(el, dim=-1, keepdim=True)
            p0 = el / z0

            
            entropy_loss2 = -torch.sum((p0 + 1e-9) * torch.log(p0 + 1e-9), dim=-1)
            onehot2 = torch.nn.functional.one_hot(torch.from_numpy(action).to(int), self.action_num).to(self.device)
            neglogpac2 = -torch.log(torch.sum(p0 * onehot2, dim=-1) + 1e-9)
            '''
            return None , neglogpac, entropy_loss, v, action_probs
            # return None , neglogpac, entropy_loss, None, action_probs
