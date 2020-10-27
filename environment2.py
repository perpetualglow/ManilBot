from copy import copy, deepcopy
import itertools
import pickle
import random

from manil.util import *
from manil.game2 import Manil2
import config as config
from manil.game import Manil, VectorizedManil
from agents.best_response import BRAgent
from agents.random_agent import RandomAgent
from agents.open_best_response import BestResponsePolicy
from agents.ddpg.maddpg import MA_DDPG
from agents.ddpg.replaybuffer import ReplayBuffer
from agents.ppo.ppo import PPO
from agents.ppo.ppo_agent import PPOAgent
from agents.dqn.dqn import DQNAgent
from manil.two_player_manil import Manil_2p
from agents.util import reorganize
from agents.polgrads import losses
import tensorflow.compat.v1 as tf
from agents.polgrads import policy_gradient

class Environment2:

    def __init__(self, game, vectorized_game):
        self.game = game
        self.vectorized_game = vectorized_game
        self.agents = []
        self.networks = []
        self.eval_agents = []
        self.state_shape = game.get_state_shape()
        self.perfect_state_shape = game.get_perfect_state_shape()
        self.old_agents = []

    def set_agents(self, agents):
        self.agents = agents

    def set_eval_agents(self, agents):
        self.eval_agents = agents

    def simulate_against_random(self, games_num):
        scores = []
        saved_games = []
        for i in range(games_num):
            player_id, state, legal_actions = self.game.start_game()
            while not self.game.is_over():
                if player_id % 2 == 0:
                    action, _ = self.eval_agents[player_id].eval_step({'obs': state, 'legal_actions': self.game.get_legal_actions_indices()})
                else:
                    action, _ = self.eval_agents[player_id].eval_step(state, legal_actions)
                self.game.step(action)
                next_player_id, next_state, legal_actions = self.game.get_state(None)
                state    = next_state
                player_id = next_player_id
            payoffs = self.game.get_payoffs()
            scores.append(payoffs[0])
        return sum(scores) / len(scores)

    def run_nfsp(self):
        trajectories = [[] for _ in range(config.num_players)]
        player_id, state, _ = self.game.start_game()
        s = {"obs": state, "legal_actions": self.game.get_legal_actions_indices()}
        trajectories[player_id].append(s)
        for i in range(config.num_rounds):
            for j in range(config.num_players):
                action = self.agents[player_id].step(s)
                reward, done, _ = self.game.step(action)
                trajectories[player_id].append(action)
                player_id, state, _ = self.game.get_state()
                s = {"obs": state, "legal_actions": self.game.get_legal_actions_indices()}
                if not self.game.is_over():
                    trajectories[player_id].append(s)
        for id in range(config.num_players):
            player_id, state, _ = self.game.get_state(id)
            trajectories[id].append({"obs": state, "legal_actions": self.game.get_legal_actions_indices()})
        payoffs = self.game.get_payoffs()
        trajectories = reorganize(trajectories, payoffs)
        for ts in trajectories[0]:
            self.agents[0].feed(ts)
        for ts in trajectories[1]:
            self.agents[1].feed(ts)

    def run_policy(self):
        random_agent = RandomAgent(config.num_cards * 2)
        with tf.Session() as sess:
            agents = [policy_gradient.PolicyGradient(
                sess,
                player_id=0,
                info_state_size=config.state_shape_2p_2,
                num_actions=config.num_cards * 2,
                loss_str="rpg",
                hidden_layers_sizes=[256, 256],
                batch_size=4,
                entropy_cost=0.001,
                critic_learning_rate=0.001,
                pi_learning_rate=0.001,
                num_critic_before_pi=128),
                policy_gradient.PolicyGradient(
                    sess,
                    player_id=1,
                    info_state_size=config.state_shape_2p_2,
                    num_actions=config.num_cards * 2,
                    loss_str="rpg",
                    hidden_layers_sizes=[256, 256],
                    batch_size=4,
                    entropy_cost=0.001,
                    critic_learning_rate=0.001,
                    pi_learning_rate=0.001,
                    num_critic_before_pi=128)
            ]
            sess.run(tf.global_variables_initializer())
            for ep in range(10000000):
                player_id, state, legal_actions = self.game.start_game()
                time_step = self.game.get_timestep()
                while not self.game.is_over():
                    act, _ = agents[player_id].step(time_step)
                    # action, _ = random_agent.step_1(legal_actions=legal_actions)
                    reward, done, _ = self.game.step(act)
                    player_id, state, legal_actions = self.game.get_state(None)
                    time_step = self.game.get_timestep()
                self.game.finish_game()
                time_step = self.game.get_timestep(0)
                agents[0].step(time_step)
                time_step = self.game.get_timestep(1)
                agents[1].step(time_step)

                if ep % 10000 == 0 and ep > 0:
                    x = 0.0
                    for i in range(5000):
                        player_id, state, legal_actions = self.game.start_game()
                        time_step = self.game.get_timestep()
                        while not self.game.is_over():
                            if player_id == 0:
                                act, _ = agents[player_id].step(time_step, is_evaluation=True)
                            else:
                                act, _ = random_agent.step_1(legal_actions=legal_actions)
                            reward, done, _ = self.game.step(act)
                            player_id, state, legal_actions = self.game.get_state(None)
                            time_step = self.game.get_timestep()
                        x += self.game.get_payoffs()[0]
                    print("Evaluation ", ep, " Score: ", x / 5000)