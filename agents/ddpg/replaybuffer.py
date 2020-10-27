"""
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

# Import Required Packages
import torch
import numpy as np
import random
from collections import namedtuple, deque

from agents.ppo.config import *

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=10000000)  # internal memory (deque)
        self.batch_size = BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["obs", "perfect_state", "legals",
                                                                "action", "reward", "next_perfect_state",
                                                                "done"])

    def add(self, obs, perfect_state, legals, action, reward, next_perfect_state, done):
        """Add a new experience to memory."""
        e = self.experience(obs, perfect_state, legals, action, reward, next_perfect_state, done)
        self.memory.append(e)

    def sample(self, agent, batch_size):
        obs_batch = []
        perfect_state_obs_batch = []
        perfect_state_batch = []
        action_batch = []
        reward_batch = []
        next_perfect_state_batch = []
        legals_batch = []
        done_batch = []
        batch = random.sample(self.memory, batch_size)
        for exp in batch:
            obs_batch.append(exp.obs[agent])
            perfect_state_batch.append(exp.perfect_state)
            perfect_obs = exp.perfect_state.encode_state_perfect(0)
            perfect_state_obs_batch.append(torch.from_numpy(perfect_obs.copy()))
            reward_batch.append(exp.reward[agent])
            action_batch.append(torch.cat(exp.action, dim=-1))
            next_perfect_state_batch.append(exp.next_perfect_state)
            done_batch.append(exp.done[agent])
            legals_batch.append(exp.legals[agent])
        return obs_batch, perfect_state_obs_batch, \
               perfect_state_batch, legals_batch, action_batch, reward_batch, next_perfect_state_batch, done_batch


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)