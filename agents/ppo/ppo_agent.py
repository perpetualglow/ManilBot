from torch.optim import Adam, RMSprop
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import random
import os
import pickle
import torch.nn.functional as F

from agents.ppo.config import *

class PPOAgent(object):

    def __init__(self, PPO, PPO_old=None, writer=None):
        self.PPO = PPO
        self.PPO_old = PPO_old
        self.learning_rate = LEARNING_RATE
        self.epsilon = EPSILON
        self.ppo_clip = PPO_CLIP
        self.gradient_clip = GRADIENT_CLIP
        self.entropy_coeff = ENTROPY_COEFFICIENT
        self.optimizer = Adam(self.PPO.parameters(True), lr=self.learning_rate, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        #self.optimizer = RMSprop(self.PPO.parameters(True), lr=self.learning_rate)
        self.value_loss = []
        self.policy_loss = []
        self.device = DEVICE
        self.clip_value = CLIP_VALUE
        self.Mseloss = nn.MSELoss()
        self.vf_coeff = VF_COEFF
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.beta = 0.01
        self.target = 0.0015
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.kl_losses = []
        self.episodes = 0
        self.writer = writer

    def calc_kl(self, p, q, get_mean=True):
        p, q = p.squeeze(), q.squeeze()
        kl = (p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum(-1)
        if get_mean:
            return kl.mean()
        return kl

    def step(self, states, legal_actions=None):
        return self.PPO(states, action=None, legal_actions=legal_actions)

    def step_value(self, states):
        return self.PPO.value(states)

    def eval_step(self, state, legal_actions):
        return self.PPO.eval_step(state, legal_actions)

    def action_probs(self, state, encoded_legal_actions, legal_actions):
        action, _, _, _, action_probs= self.PPO.eval_step(state, encoded_legal_actions)
        #action = action.item()
        res = []
        for i,p in enumerate(action_probs[0]):
            if p != 0:
                res.append((i,p))
        return res
        #return [(a, prob) for a in legal_actions]

    def action_probabilities(self, state):
        legal_actions = state.encode_legal_actions()
        _, _, _, _, action_probs = self.PPO.eval_step([state.encode_state(None)], legal_actions)
        return {act: prob for act, prob in enumerate(action_probs[0]) if prob != 0}


    def run(self, states, legal_actions, returns, actions, values, advantages, logprobs, probs):
        l_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        l_advantages = torch.from_numpy(l_advantages).to(self.device)
        l_returns = torch.from_numpy(returns).to(self.device)
        l_logprobs = torch.from_numpy(logprobs).to(self.device)
        l_values = torch.from_numpy(values).to(self.device)
        l_probs = torch.from_numpy(probs).to(self.device)
        l_actions = torch.from_numpy(actions).to(self.device)
        valloss = 0
        polloss = 0
        entrloss = 0
        totalloss = 0
        n_its = self.epochs * (len(actions) // self.batch_size)
        for i in range(self.epochs):
            indecies = []
            rl = list(range(0, len(actions)))
            random.shuffle(rl)
            for num in range(0, len(rl) // self.batch_size):
                indecies.append(rl[num * self.batch_size:num * self.batch_size + self.batch_size])
            for batch_indices in indecies:
                sampled_states = states[batch_indices]
                # sampled_perfect_states = perfect_states[batch_indices]
                sampled_legal_actions = legal_actions[batch_indices]
                sampled_actions = l_actions[batch_indices]
                sampled_log_probs_old = l_logprobs[batch_indices]
                sampled_returns = l_returns[batch_indices]
                sampled_advantages = l_advantages[batch_indices]
                sampled_values = l_values[batch_indices]
                sampled_probs = l_probs[batch_indices]
                # _, r_log_probs, r_entropy_loss, r_values, prob = self.PPO(sampled_states, sampled_actions, sampled_legal_actions)
                _, r_log_probs, r_entropy_loss, r_values, prob = self.PPO(sampled_states, sampled_actions,
                                                                          sampled_legal_actions)
                # r_values = self.PPO.value_ppo(sampled_perfect_states)
                r_values = torch.flatten(r_values)
                if self.clip_value:
                    value_pred_clipped = sampled_values + \
                                         (r_values - sampled_values).clamp(-self.ppo_clip, self.ppo_clip)
                    value_losses = (r_values - sampled_returns).pow(2)
                    value_losses_clipped = (value_pred_clipped - sampled_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    predicted = r_values.to(self.device)
                    target = sampled_returns.type(predicted.type()).to(predicted.device)
                    value_loss = self.Mseloss(predicted, target)
                # value_loss = F.smooth_l1_loss(r_values.float(), sampled_returns.float()).float()
                #x = -torch.log(sampled_probs + 1e-9)
                #out = torch.nn.functional.kl_div(-torch.log(sampled_probs + 1e-9), -torch.log(prob + 1e-9), reduction='batchmean', log_target=True)
                # kl_penalty = self.calc_kl(prob, sampled_probs)
                ratio = (-sampled_log_probs_old + r_log_probs).exp()
                p_losses1 = sampled_advantages * ratio
                p_losses2 = ratio.clamp(1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * sampled_advantages
                policy_loss = -torch.min(p_losses1, p_losses2).mean().float()
                # print("Policy loss ", policy_loss)
                entropy_loss = r_entropy_loss.mean()
                # valloss += value_loss.item()
                # polloss += -policy_loss.item()
                # entrloss += entropy_loss
                #
                # self.entropy_losses.append(entropy_loss.item())
                # self.value_losses.append(value_loss.item())
                # self.kl_losses.append(kl_penalty.item())
                # total_loss = (policy_loss + value_loss)
                total_loss = (policy_loss + value_loss - 0.01 * entropy_loss)
                # total_loss = (policy_loss + 0.1 * value_loss)
                # totalloss += total_loss.item()
                #print("Value loss ", value_loss)
                self.episodes += 1
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.PPO.parameters(), self.gradient_clip)
                self.optimizer.step()
        self.scheduler.step()
        # self.writer.add_scalar('Loss/Total_loss', totalloss / n_its, self.episodes)
        # self.writer.add_scalar('Loss/Policy_loss', polloss / n_its, self.episodes)
        # self.writer.add_scalar('Loss/Value_loss', valloss / n_its, self.episodes)
        # self.writer.add_scalar('Loss/Entropy_loss', entrloss / n_its, self.episodes)
        self.episodes += 1
        reset = False
        for param in self.PPO.parameters():
            if np.sum(np.isnan(param.cpu().detach().numpy())) > 0:
                reset = True

        if reset:
            print('reset')
            load_path = os.path.join('models', 'cpu_1536_768_256_64')
            self.PPO.load_state_dict(torch.load(load_path))

    def save(self, path):
        '''
        agent_path = path + "_agent.pkl"
        with open(agent_path, 'wb') as file_handler:
            pickle.dump(self, file_handler)
        '''
        name = path + "_ntk.pth"
        torch.save(self.PPO.state_dict(), name)

    def load(self, path):
        network_path = path + "_ntk.pth"
        self.PPO.load_state_dict(torch.load(network_path, map_location=torch.device('cpu')))
        self.PPO.to(self.device)


