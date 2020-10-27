import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from agents.ddpg.ddpg import Actor, Critic
from agents.ppo.config import *


class Single_DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_shape, num_actions, num_agents, index):
        self.device = DEVICE
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.index = index


