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

class Environment:

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

    def update_agents(self):
        self.agents[0].save("temp")
        network_2p = PPO(self.game.get_state_shape(), self.game.get_state_shape(), config.num_cards * 2, config.network_shape)
        old_agent = PPOAgent(network_2p)
        old_agent.load("temp")
        self.old_agents.append(old_agent)
        r = random.randint(0,len(self.old_agents) - 1)
        return self.old_agents[r]

    def simulate_ddpg_against_random(self, games_num):
        scores = []
        saved_games = []
        for i in range(games_num):
            player_id, state, legal_actions = self.game.start_game()
            while not self.game.is_over():
                if player_id % 2 == 0:
                    action = self.eval_agents[player_id].sim_step(state, legal_actions, len(self.game.current_trick))
                else:
                    action, _ = self.eval_agents[player_id].eval_step(state, legal_actions)
                self.game.step(action.item())
                next_player_id, next_state, legal_actions = self.game.get_state(None)
                state = next_state
                player_id = next_player_id
            payoffs = self.game.get_payoffs()
            scores.append(payoffs[0])
        return sum(scores) / len(scores)

    def simulate_against_random(self, games_num):
        scores = []
        saved_games = []
        for i in range(games_num):
            player_id, state, legal_actions = self.game.start_game()
            while not self.game.is_over():
                if player_id % 2 == 0:
                    action, _, _, _, _ = self.eval_agents[0].eval_step([state], legal_actions)
                else:
                    action, _ = self.eval_agents[1].eval_step(state, legal_actions)
                self.game.step(action.item())
                next_player_id, next_state, legal_actions = self.game.get_state(None)
                state = next_state
                player_id = next_player_id
            payoffs = self.game.get_payoffs()
            scores.append(payoffs[0])
        return sum(scores) / len(scores)

    def simulate_against_random2(self, games_num):
        scores = []
        saved_games = []
        game = Manil(eval=True)
        wrongs = 0.0
        for i in range(games_num):
            player_id, state, legal_actions = game.start_game2()
            while not game.is_over():
                if player_id == 0:
                    action, _, _, _, _ = self.eval_agents[player_id].eval_step([state], legal_actions)
                else:
                    action, _ = self.eval_agents[1].eval_step2(state, legal_actions)
                _, _, _, wrong = game.step2(action.item())
                if wrong:
                    wrongs += 1
                next_player_id, next_state, legal_actions = game.get_state2(None)
                state = next_state
                player_id = next_player_id
            payoffs = game.get_payoffs()
            scores.append(payoffs[0])
        print("Number wrongs: ", wrongs / (games_num * 2))
        return sum(scores) / len(scores)


    def run_dqn(self):
        game = Manil_2p()
        agent = DQNAgent(scope='dqn',
                         action_num=config.num_cards * 2,
                         replay_memory_init_size=1000,
                         train_every=10,
                         state_shape=self.game.get_state_shape(),
                         mlp_layers=[256,256],
                         device=torch.device('cpu'))
        agents = [DQNAgent(scope='dqn',
                         action_num=config.num_cards * 2,
                         replay_memory_init_size=1000,
                         train_every=10,
                         state_shape=self.game.get_state_shape(),
                         mlp_layers=[256,256],
                         device=torch.device('cpu')),
                  DQNAgent(scope='dqn',
                         action_num=config.num_cards * 2,
                         replay_memory_init_size=1000,
                         train_every=10,
                         state_shape=self.game.get_state_shape(),
                         mlp_layers=[256,256],
                         device=torch.device('cpu'))]
        for g in range(10000):
            trajectories = [[] for _ in range(config.num_players)]
            player_id, state, _ = game.start_game()
            s = {"obs": state, "legal_actions": game.get_legal_actions_indices()}
            trajectories[player_id].append(s)
            for i in range(config.num_rounds):
                for j in range(config.num_players):
                    action = agents[player_id].step(s)
                    reward, done, _ = game.step(action)
                    trajectories[player_id].append(action)
                    player_id, state, _ = game.get_state()
                    s = {"obs": state, "legal_actions": game.get_legal_actions_indices()}
                    if not game.is_over():
                        trajectories[player_id].append(s)
            for id in range(config.num_players):
                player_id, state, _ = game.get_state(id)
                trajectories[id].append({"obs": state, "legal_actions": game.get_legal_actions_indices()})
            payoffs = game.get_payoffs()
            trajectories = self.reorganize(trajectories, payoffs)
            for ts in trajectories[0]:
                agents[0].feed(ts)
            for ts in trajectories[1]:
                agents[1].feed(ts)


    def br2(self, episodes, agent):
        game = Manil2()
        expl = 0.0
        for i in range(episodes):
            root_state = game.new_initial_state()
            x1 = BestResponsePolicy(game, 0, agent, root_state).value(root_state)
            x2 = BestResponsePolicy(game, 1, agent, root_state).value(root_state)
            avg = ((x1 + x2) - game.utility_sum()) / 2
            print("Average ", i, " ", avg)
            expl += avg
        print ("Exploitability ", expl / episodes)

    def full_exploitability(self, states, agent):
        game = Manil2()
        expl = 0.0
        for state in states:
            x1 = BestResponsePolicy(game, 0, agent, state).value(state)
            x2 = BestResponsePolicy(game, 1, agent, state).value(state)
            avg = ((x1 + x2) - game.utility_sum()) / 2
            expl += avg
        print("Exploitability ", expl / len(states))

    def train_alone(self):
        num_games, num_cards, num_players, num_total_games = config.NUM_CONCURRENT_GAMES, config.num_cards, config.num_players, config.NUM_TOTAL_GAMES
        shape = np.array(list(self.state_shape))
        state_shape = np.prod(shape)
        mb_state_shape = np.array([num_total_games, config.num_cards, state_shape])





    def train_against_random2(self):
        num_games, num_cards, num_players, num_total_games = config.NUM_CONCURRENT_GAMES, config.num_cards, config.num_players, config.NUM_TOTAL_GAMES
        shape = np.array(list(self.state_shape))
        state_shape = np.prod(shape)
        mb_state_shape = np.array([num_total_games, config.num_cards, state_shape])
        mb_states, mb_ids, mb_actions, mb_values, mb_logprobs, mb_rewards, mb_dones, mb_legal_actions, mb_player_ids, mb_probs = np.zeros(
            tuple(mb_state_shape)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 config.num_cards)), np.zeros(
            (num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards + config.num_players)), np.zeros(
            (num_total_games, num_cards)), np.zeros((num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards)), np.zeros(
            (num_total_games, num_cards, num_cards)), np.zeros((num_total_games, num_cards)), np.zeros((num_total_games, num_cards, num_cards))
        for x in range(num_total_games // num_games):
            player_ids, states, legal_actions = self.vectorized_game.reset2()
            for i in range(num_cards):
                ind_rand = np.where(player_ids != 0)[0]
                ind_ppo = np.where(player_ids == 0)[0]
                actions = np.zeros(len(player_ids), dtype=int)
                values_tot = np.zeros(len(player_ids))
                log_probs_tot = np.zeros(len(player_ids))
                probs_tot = np.zeros((len(player_ids), num_cards))
                if len(ind_rand) > 0:
                    actions_rand, _ = self.agents[1].step(legal_actions[ind_rand])
                    actions[ind_rand] = actions_rand
                if len(ind_ppo) > 0:
                    actions_ppo, log_probs, _, values, probs = self.agents[0].step(states[ind_ppo], legal_actions=legal_actions[ind_ppo])
                    actions[ind_ppo] = actions_ppo
                    log_probs_tot[ind_ppo] = log_probs
                    values_tot[ind_ppo] = values.flatten()
                    probs_tot[ind_ppo] = probs
                played_actions, rewards, dones, infos = self.vectorized_game.step2(actions)
                round_num = i // num_players
                for game in range(len(player_ids)):
                    g = x * num_games + game
                    mb_states[g][round_num * num_players + player_ids[game]] = states[game].copy()
                    mb_actions[g][round_num * num_players + player_ids[game]] = actions[game]
                    mb_values[g][round_num * num_players + player_ids[game]] = values_tot[game]
                    mb_logprobs[g][round_num * num_players + player_ids[game]] = log_probs_tot[game]
                    mb_dones[g][round_num * num_players + player_ids[game]] = dones[game]
                    mb_rewards[g][round_num * num_players + player_ids[game]] = rewards[game]
                    mb_legal_actions[g][round_num * num_players + player_ids[game]] = legal_actions[game].copy()
                    mb_player_ids[g][round_num * num_players + player_ids[game]] = player_ids[game]
                    mb_probs[g][round_num * num_players + player_ids[game]] = probs_tot[game]
                player_ids, states, legal_actions = self.vectorized_game.get_state2(None)
            '''
            for i in range(config.num_players):
                t_player_ids, t_states, t_legal_actions = self.vectorized_game.get_endstate(i)
                _, _, _, t_values, _ = self.agents[0].step(t_states)
                for g in range(len(t_values)):
                    mb_values[x * num_games + g][num_cards + i] = t_values[g][0]
            '''
            reward = self.vectorized_game.get_payoffs()
            reward = reward.swapaxes(1,0)
            for g in range(num_games):
                mb_rewards[x * num_games + g][num_cards - config.num_players:num_cards] = reward[g]
                for i in range(config.num_players):
                    mb_values[x * num_games + g][num_cards + i] = reward[g][i]
        mb_states_2 = mb_states.swapaxes(1, 0)
        mb_values_2 = mb_values.swapaxes(1,0)
        mb_logprobs_2 = mb_logprobs.swapaxes(1,0)
        mb_dones_2 = mb_dones.swapaxes(1,0)
        mb_rewards_2 = mb_rewards.swapaxes(1,0)
        mb_legal_actions_2 = mb_legal_actions.swapaxes(1,0)
        mb_actions_2 = mb_actions.swapaxes(1,0)
        mb_player_ids_2 = mb_player_ids.swapaxes(1,0)
        mb_probs_2 = mb_probs.swapaxes(1,0)
        mb_returns = np.zeros_like(mb_rewards_2)
        mb_advs = np.zeros_like(mb_rewards_2)
        for k in range(config.num_players):
            lastgaelam = 0
            for t in reversed(range(k, num_cards, config.num_players)):
                nextNonTerminal = mb_dones_2[t]
                nextValues = mb_values_2[t + config.num_players]
                delta = mb_rewards_2[t] + 0.99 * nextValues * nextNonTerminal - mb_values_2[t]
                mb_advs[t] = lastgaelam = delta + 0.99 * 0.99 * nextNonTerminal * lastgaelam
        mb_values_3 = mb_values_2[0:num_cards]
        mb_returns = mb_advs + mb_values_3
        statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, player_idsr, probsr = map(self.sf01,
                                                                                    (mb_states_2, mb_legal_actions_2,
                                                                                     mb_returns, mb_actions_2, mb_values_3,
                                                                                     mb_advs, mb_logprobs_2, mb_player_ids_2,
                                                                                     mb_probs_2))

        indices = np.where(player_idsr % 2 == 0)[0]
        statesr = statesr[indices]
        legal_actionsr = legal_actionsr[indices]
        returnsr = returnsr[indices]
        actionsr = actionsr[indices]
        valuesr = valuesr[indices]
        advantagesr = advantagesr[indices]
        logprobsr = logprobsr[indices]
        probsr = probsr[indices]

        self.agents[0].run(statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, probsr)

    def train_against_self2(self):
        num_games, num_cards, num_players, num_total_games = config.NUM_CONCURRENT_GAMES, config.num_cards, config.num_players, config.NUM_TOTAL_GAMES
        shape = np.array(list(self.state_shape))
        perfect_shape = np.array(list(self.perfect_state_shape))
        state_shape = np.prod(shape)
        perfect_state_shape = np.prod(perfect_shape)
        mb_state_shape = np.array([num_total_games, config.num_cards, state_shape])
        mb_perfect_state_shape = np.array([num_total_games, config.num_cards, perfect_state_shape])
        mb_states, mb_perfect_states, mb_ids, mb_actions, mb_values, mb_logprobs, mb_rewards, mb_dones, mb_legal_actions, mb_player_ids, mb_probs = np.zeros(
            tuple(mb_state_shape)), np.zeros(tuple(mb_perfect_state_shape)),\
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 config.num_cards)), np.zeros(
            (num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards + 2)), np.zeros(
            (num_total_games, num_cards)), np.zeros((num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards)), np.zeros(
            (num_total_games, num_cards, num_cards)), np.zeros((num_total_games, num_cards)), np.zeros((num_total_games, num_cards, num_cards))
        for x in range(num_total_games // num_games):
            player_ids, states, legal_actions = self.vectorized_game.reset()
            _ , perfect_states, _ = self.vectorized_game.get_perfect_state()
            for i in range(num_cards):
                actions, log_probs, _, _, probs = self.agents[0].step(states, legal_actions)
                values = self.agents[0].step_value(perfect_states)
                rewards, dones, _ = self.vectorized_game.step(actions)
                round_num = i // num_players
                if i % 4 >= 2:
                    y = 2
                else:
                    y = 0
                for game in range(len(player_ids)):
                    z = player_ids[game] % 2 + y
                    g = x * num_games + game
                    mb_states[g][round_num * num_players + z] = states[game].copy()
                    mb_perfect_states[g][round_num * num_players + z] = perfect_states[game].copy()
                    mb_actions[g][round_num * num_players + z] = actions[game]
                    mb_values[g][round_num * num_players + z] = values[game][0]
                    mb_logprobs[g][round_num * num_players + z] = log_probs[game]
                    mb_dones[g][round_num * num_players + z] = dones[game]
                    mb_rewards[g][round_num * num_players + z] = rewards[game]
                    mb_legal_actions[g][round_num * num_players + z] = legal_actions[game].copy()
                    mb_player_ids[g][round_num * num_players + z] = player_ids[game]
                    mb_probs[g][round_num * num_players + z] = probs[game]
                player_ids, states, legal_actions = self.vectorized_game.get_state(None)
                _, perfect_states, _ = self.vectorized_game.get_perfect_state()

            for i in range(2):
                t_player_ids, t_states, t_legal_actions = self.vectorized_game.get_perfect_state(i)
                t_values = self.agents[0].step_value(t_states)
                for g in range(len(t_values)):
                    mb_values[x * num_games + g][num_cards + i] = t_values[g][0]
            reward = self.vectorized_game.get_payoffs()
            reward = reward.swapaxes(1,0)
            for g in range(num_games):
                mb_rewards[x * num_games + g][num_cards - 2] = reward[g][0]
                mb_rewards[x * num_games + g][num_cards - 1] = reward[g][1]
        mb_states_2 = mb_states.swapaxes(1, 0)
        mb_perfect_states_2 = mb_perfect_states.swapaxes(1,0)
        mb_values_2 = mb_values.swapaxes(1,0)
        mb_logprobs_2 = mb_logprobs.swapaxes(1,0)
        mb_dones_2 = mb_dones.swapaxes(1,0)
        mb_rewards_2 = mb_rewards.swapaxes(1,0)
        mb_legal_actions_2 = mb_legal_actions.swapaxes(1,0)
        mb_actions_2 = mb_actions.swapaxes(1,0)
        mb_player_ids_2 = mb_player_ids.swapaxes(1,0)
        mb_probs_2 = mb_probs.swapaxes(1,0)
        mb_returns = np.zeros_like(mb_rewards_2)
        mb_returns_2 = np.zeros_like(mb_rewards_2)
        mb_advs = np.zeros_like(mb_rewards_2)

        for k in range(2):
            lastgaelam = 0
            for t in reversed(range(k, num_cards, 2)):
                nextNonTerminal = mb_dones_2[t]
                nextValues = mb_values_2[t + 2]
                delta = mb_rewards_2[t] + 0.99 * nextValues * nextNonTerminal - mb_values_2[t]
                mb_advs[t] = lastgaelam = delta + 0.99 * 0.99 * nextNonTerminal * lastgaelam
        mb_values_3 = mb_values_2[0:num_cards]
        mb_returns = mb_advs + mb_values_3

        statesr, perfect_statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, player_idsr, probsr = map(self.sf01,
                                                                                    (mb_states_2, mb_perfect_states_2, mb_legal_actions_2,
                                                                                     mb_returns, mb_actions_2, mb_values_3,
                                                                                     mb_advs, mb_logprobs_2, mb_player_ids_2,
                                                                                     mb_probs_2))

        self.agents[0].run(statesr, perfect_statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, probsr)

    def train_against_self(self):
        num_games, num_cards, num_players, num_total_games = config.NUM_CONCURRENT_GAMES, config.num_cards, config.num_players, config.NUM_TOTAL_GAMES
        mb_state_shape = copy(list(self.state_shape))
        mb_state_shape.insert(0, num_cards)
        mb_state_shape.insert(0, num_total_games)
        mb_states, mb_ids, mb_actions, mb_values, mb_logprobs, mb_rewards, mb_dones, mb_legal_actions, mb_player_ids, mb_probs = np.zeros(
            tuple(mb_state_shape)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 config.num_cards)), np.zeros(
            (num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards + config.num_players)), np.zeros(
            (num_total_games, num_cards)), np.zeros((num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards)), np.zeros(
            (num_total_games, num_cards, num_cards*2)), np.zeros((num_total_games, num_cards)), np.zeros((num_total_games, num_cards, 2*num_cards))
        for x in range(num_total_games // num_games):
            player_ids, states, legal_actions = self.vectorized_game.reset()
            for i in range(num_cards):
                actions, log_probs, _, values, probs = self.agents[0].step(states, legal_actions)
                rewards, dones, infos = self.vectorized_game.step(actions)
                round_num = i // num_players
                for game in range(len(player_ids)):
                    g = x * num_games + game
                    mb_states[g][round_num * num_players + player_ids[game]] = states[game].copy()
                    mb_actions[g][round_num * num_players + player_ids[game]] = actions[game]
                    mb_values[g][round_num * num_players + player_ids[game]] = values[game][0]
                    mb_logprobs[g][round_num * num_players + player_ids[game]] = log_probs[game]
                    mb_dones[g][round_num * num_players + player_ids[game]] = dones[game]
                    mb_rewards[g][round_num * num_players + player_ids[game]] = rewards[game]
                    mb_legal_actions[g][round_num * num_players + player_ids[game]] = legal_actions[game].copy()
                    mb_player_ids[g][round_num * num_players + player_ids[game]] = player_ids[game]
                    mb_probs[g][round_num * num_players + player_ids[game]] = probs[game]
                player_ids, states, legal_actions = self.vectorized_game.get_state(None)
            for i in range(config.num_players):
                t_player_ids, t_states, t_legal_actions = self.vectorized_game.get_state(i)
                _, _, _, t_values, _ = self.agents[0].step(t_states, legal_actions=None)
                for g in range(len(t_values)):
                    mb_values[x * num_games + g][num_cards + i] = t_values[g][0]
            reward = self.vectorized_game.get_payoffs()
            reward = reward.swapaxes(1,0)
            for g in range(num_games):
                mb_rewards[x * num_games + g][num_cards - config.num_players:num_cards] = reward[g]
        mb_states_2 = mb_states.swapaxes(1, 0)
        mb_values_2 = mb_values.swapaxes(1,0)
        mb_logprobs_2 = mb_logprobs.swapaxes(1,0)
        mb_dones_2 = mb_dones.swapaxes(1,0)
        mb_rewards_2 = mb_rewards.swapaxes(1,0)
        mb_legal_actions_2 = mb_legal_actions.swapaxes(1,0)
        mb_actions_2 = mb_actions.swapaxes(1,0)
        mb_player_ids_2 = mb_player_ids.swapaxes(1,0)
        mb_probs_2 = mb_probs.swapaxes(1,0)
        mb_returns = np.zeros_like(mb_rewards_2)
        mb_advs = np.zeros_like(mb_rewards_2)
        for k in range(config.num_players):
            lastgaelam = 0
            for t in reversed(range(k, num_cards, config.num_players)):
                nextNonTerminal = mb_dones_2[t]
                nextValues = mb_values_2[t + config.num_players]
                delta = mb_rewards_2[t] + 0.99 * nextValues * nextNonTerminal - mb_values_2[t]
                mb_advs[t] = lastgaelam = delta + 0.99 * 0.99 * nextNonTerminal * lastgaelam
        mb_values_3 = mb_values_2[0:num_cards]
        mb_returns = mb_advs + mb_values_3
        statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, player_idsr, probsr = map(self.sf01,
                                                                                    (mb_states_2, mb_legal_actions_2,
                                                                                     mb_returns, mb_actions_2, mb_values_3,
                                                                                     mb_advs, mb_logprobs_2, mb_player_ids_2,
                                                                                     mb_probs_2))

        self.agents[0].run(statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, probsr)


    def train_against_old(self, agent=None):
        if agent is None:
            agent = self.agents[1]
        num_games, num_cards, num_players, num_total_games = config.NUM_CONCURRENT_GAMES, config.num_cards, config.num_players, config.NUM_TOTAL_GAMES
        mb_state_shape = copy(list(self.state_shape))
        mb_state_shape.insert(0, num_cards)
        mb_state_shape.insert(0, num_total_games)
        mb_states, mb_ids, mb_actions, mb_values, mb_logprobs, mb_rewards, mb_dones, mb_legal_actions, mb_player_ids, mb_probs = np.zeros(
            tuple(mb_state_shape)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 config.num_cards)), np.zeros(
            (num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards + config.num_players)), np.zeros(
            (num_total_games, num_cards)), np.zeros((num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards)), np.zeros(
            (num_total_games, num_cards, num_cards*2)), np.zeros((num_total_games, num_cards)), np.zeros((num_total_games, num_cards, 2*num_cards))
        for x in range(num_total_games // num_games):
            player_ids, states, legal_actions = self.vectorized_game.reset()
            for i in range(num_cards):
                ind_rand = np.where(player_ids != 0)[0]
                ind_ppo = np.where(player_ids == 0)[0]
                actions = np.zeros(len(player_ids), dtype=int)
                values_tot = np.zeros(len(player_ids))
                log_probs_tot = np.zeros(len(player_ids))
                probs_tot = np.zeros((len(player_ids), num_cards*2))
                if len(ind_rand) > 0:
                    actions_rand, _ , _, _, _= agent.eval_step(states[ind_rand], legal_actions=legal_actions[ind_rand])
                    actions[ind_rand] = actions_rand
                if len(ind_ppo) > 0:
                    actions_ppo, log_probs, _, values, probs = self.agents[0].step(states[ind_ppo], legal_actions=legal_actions[ind_ppo])
                    actions[ind_ppo] = actions_ppo
                    log_probs_tot[ind_ppo] = log_probs
                    values_tot[ind_ppo] = values.flatten()
                    probs_tot[ind_ppo] = probs
                played_actions, rewards, dones = self.vectorized_game.step(actions)
                round_num = i // num_players
                for game in range(len(player_ids)):
                    g = x * num_games + game
                    mb_states[g][round_num * num_players + player_ids[game]] = states[game].copy()
                    mb_actions[g][round_num * num_players + player_ids[game]] = actions[game]
                    mb_values[g][round_num * num_players + player_ids[game]] = values_tot[game]
                    mb_logprobs[g][round_num * num_players + player_ids[game]] = log_probs_tot[game]
                    mb_dones[g][round_num * num_players + player_ids[game]] = dones[game]
                    mb_rewards[g][round_num * num_players + player_ids[game]] = rewards[game]
                    mb_legal_actions[g][round_num * num_players + player_ids[game]] = legal_actions[game].copy()
                    mb_player_ids[g][round_num * num_players + player_ids[game]] = player_ids[game]
                    mb_probs[g][round_num * num_players + player_ids[game]] = probs_tot[game]
                player_ids, states, legal_actions = self.vectorized_game.get_state(None)

            for i in range(config.num_players):
                t_player_ids, t_states, t_legal_actions = self.vectorized_game.get_state(i)
                _, _, _, t_values, _ = self.agents[0].step(t_states)
                for g in range(len(t_values)):
                    mb_values[x * num_games + g][num_cards + i] = t_values[g][0]
            reward = self.vectorized_game.get_payoffs()
            reward = reward.swapaxes(1,0)
            for g in range(num_games):
                mb_rewards[x * num_games + g][num_cards - config.num_players:num_cards] = reward[g]
        mb_states_2 = mb_states.swapaxes(1, 0)
        mb_values_2 = mb_values.swapaxes(1,0)
        mb_logprobs_2 = mb_logprobs.swapaxes(1,0)
        mb_dones_2 = mb_dones.swapaxes(1,0)
        mb_rewards_2 = mb_rewards.swapaxes(1,0)
        mb_legal_actions_2 = mb_legal_actions.swapaxes(1,0)
        mb_actions_2 = mb_actions.swapaxes(1,0)
        mb_player_ids_2 = mb_player_ids.swapaxes(1,0)
        mb_probs_2 = mb_probs.swapaxes(1,0)
        mb_returns = np.zeros_like(mb_rewards_2)
        mb_advs = np.zeros_like(mb_rewards_2)
        for k in range(config.num_players):
            lastgaelam = 0
            for t in reversed(range(k, num_cards, config.num_players)):
                nextNonTerminal = mb_dones_2[t]
                nextValues = mb_values_2[t + config.num_players]
                delta = mb_rewards_2[t] + 0.99 * nextValues * nextNonTerminal - mb_values_2[t]
                mb_advs[t] = lastgaelam = delta + 0.99 * 0.99 * nextNonTerminal * lastgaelam
        mb_values_3 = mb_values_2[0:num_cards]
        mb_returns = mb_advs + mb_values_3
        statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, player_idsr, probsr = map(self.sf01,
                                                                                    (mb_states_2, mb_legal_actions_2,
                                                                                     mb_returns, mb_actions_2, mb_values_3,
                                                                                     mb_advs, mb_logprobs_2, mb_player_ids_2,
                                                                                     mb_probs_2))

        indices = np.where(player_idsr % 2 == 0)[0]
        statesr = statesr[indices]
        legal_actionsr = legal_actionsr[indices]
        returnsr = returnsr[indices]
        actionsr = actionsr[indices]
        valuesr = valuesr[indices]
        advantagesr = advantagesr[indices]
        logprobsr = logprobsr[indices]
        probsr = probsr[indices]

        self.agents[0].run(statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, probsr)

    def multi_train_against_self(self):
        num_games, num_cards, num_players, num_total_games = config.NUM_CONCURRENT_GAMES, config.num_cards, config.num_players, config.NUM_TOTAL_GAMES
        mb_state_shape = copy(list(self.state_shape))
        mb_state_shape.insert(0, num_cards)
        mb_state_shape.insert(0, num_total_games)
        mb_perfect_state_shape = copy(list(self.perfect_state_shape))
        mb_perfect_state_shape.insert(0, num_cards)
        mb_perfect_state_shape.insert(0, num_total_games)
        mb_states, mb_ids, mb_actions, mb_values, mb_logprobs, mb_rewards, mb_dones, mb_legal_actions, mb_player_ids, mb_probs = np.zeros(
            tuple(mb_state_shape)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 config.num_cards)), np.zeros(
            (num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards + config.num_players)), np.zeros(
            (num_total_games, num_cards)), np.zeros((num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards)), np.zeros(
            (num_total_games, num_cards, num_cards)), np.zeros((num_total_games, num_cards)), np.zeros((num_total_games, num_cards, num_cards))
        mb_perfect_states = np.zeros(tuple(mb_perfect_state_shape))
        for x in range(num_total_games // num_games):
            player_ids, states, legal_actions = self.vectorized_game.reset()
            for i in range(num_cards):
                actions, log_probs, _, values, probs = self.agents[0].step(states, legal_actions)
                rewards, dones, infos = self.vectorized_game.step(actions)
                round_num = i // num_players
                for game in range(len(player_ids)):
                    g = x * num_games + game
                    mb_states[g][round_num * num_players + player_ids[game]] = states[game].copy()
                    mb_actions[g][round_num * num_players + player_ids[game]] = actions[game]
                    #mb_values[g][round_num * num_players + player_ids[game]] = values[game][0]
                    mb_logprobs[g][round_num * num_players + player_ids[game]] = log_probs[game]
                    mb_dones[g][round_num * num_players + player_ids[game]] = dones[game]
                    mb_rewards[g][round_num * num_players + player_ids[game]] = rewards[game]
                    mb_legal_actions[g][round_num * num_players + player_ids[game]] = legal_actions[game].copy()
                    mb_player_ids[g][round_num * num_players + player_ids[game]] = player_ids[game]
                    mb_probs[g][round_num * num_players + player_ids[game]] = probs[game]
                if (i + 1) % num_players == 0:
                    for player in range(config.num_players):
                        _, perfect_states, _ = self.vectorized_game.get_perfect_state(player)
                        values = self.agents[0].step_value(perfect_states)
                        for game in range(len(perfect_states)):
                            mb_perfect_states[x * num_games + game][round_num * num_players + player] = perfect_states[game].copy()
                            mb_values[x*num_games + game][round_num * num_players + player] = values[game][0]
                player_ids, states, legal_actions = self.vectorized_game.get_state(None)
            reward = self.vectorized_game.get_payoffs()
            reward = reward.swapaxes(1, 0)
            for i in range(config.num_players):
                _, t_states, _= self.vectorized_game.get_perfect_state(i)
                t_values = self.agents[0].step_value(t_states)
                for g in range(len(t_values)):
                    mb_values[x * num_games + g][num_cards + i] = t_values[g]
            # reward = self.vectorized_game.get_payoffs()
            # reward = reward.swapaxes(1, 0)
            for g in range(num_games):
                mb_rewards[x * num_games + g][num_cards - config.num_players:num_cards] = reward[g]
        mb_states_2 = mb_states.swapaxes(1, 0)
        mb_perfect_states_2 = mb_perfect_states.swapaxes(1,0)
        mb_values_2 = mb_values.swapaxes(1,0)
        mb_logprobs_2 = mb_logprobs.swapaxes(1,0)
        mb_dones_2 = mb_dones.swapaxes(1,0)
        mb_rewards_2 = mb_rewards.swapaxes(1,0)
        mb_legal_actions_2 = mb_legal_actions.swapaxes(1,0)
        mb_actions_2 = mb_actions.swapaxes(1,0)
        mb_player_ids_2 = mb_player_ids.swapaxes(1,0)
        mb_probs_2 = mb_probs.swapaxes(1,0)
        mb_returns = np.zeros_like(mb_rewards_2)
        mb_advs = np.zeros_like(mb_rewards_2)
        for k in range(config.num_players):
            lastgaelam = 0
            for t in reversed(range(k, num_cards, config.num_players)):
                nextNonTerminal = mb_dones_2[t]
                nextValues = mb_values_2[t + config.num_players]
                delta = mb_rewards_2[t] + 0.99 * nextValues * nextNonTerminal - mb_values_2[t]
                mb_advs[t] = lastgaelam = delta + 0.99 * 0.99 * nextNonTerminal * lastgaelam
        mb_values_3 = mb_values_2[0:num_cards]
        mb_returns = mb_advs + mb_values_3
        statesr, perfect_statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, player_idsr, probsr = map(self.sf01,
                                                                                    (mb_states_2, mb_perfect_states_2,
                                                                                     mb_legal_actions_2,
                                                                                     mb_returns, mb_actions_2, mb_values_3,
                                                                                     mb_advs, mb_logprobs_2, mb_player_ids_2,
                                                                                     mb_probs_2))

        self.agents[0].run(statesr, perfect_statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, probsr)

    def multi_train_against_random(self):
        num_games, num_cards, num_players, num_total_games = config.NUM_CONCURRENT_GAMES, config.num_cards, config.num_players, config.NUM_TOTAL_GAMES
        random_agent = RandomAgent(num_cards)
        mb_state_shape = copy(list(self.state_shape))
        mb_state_shape.insert(0, num_cards)
        mb_state_shape.insert(0, num_total_games)
        mb_perfect_state_shape = copy(list(self.perfect_state_shape))
        mb_perfect_state_shape.insert(0, num_cards)
        mb_perfect_state_shape.insert(0, num_total_games)
        mb_states, mb_ids, mb_actions, mb_values, mb_logprobs, mb_rewards, mb_dones, mb_legal_actions, mb_player_ids, mb_probs = np.zeros(
            tuple(mb_state_shape)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 config.num_cards)), np.zeros(
            (num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards + config.num_players)), np.zeros(
            (num_total_games, num_cards)), np.zeros((num_total_games, num_cards)), \
                                                                                                        np.zeros((
                                                                                                                 num_total_games,
                                                                                                                 num_cards)), np.zeros(
            (num_total_games, num_cards, num_cards)), np.zeros((num_total_games, num_cards)), np.zeros((num_total_games, num_cards, num_cards))
        mb_perfect_states = np.zeros(tuple(mb_perfect_state_shape))
        for x in range(num_total_games // num_games):
            player_ids, states, legal_actions = self.vectorized_game.reset()
            for i in range(num_cards):
                ind_rand = np.where(player_ids != 0)[0]
                ind_ppo = np.where(player_ids == 0)[0]
                actions = np.zeros(len(player_ids), dtype=int)
                log_probs_tot = np.zeros(len(player_ids))
                probs_tot = np.zeros((len(player_ids), num_cards))
                if len(ind_rand) > 0:
                    actions_rand, _ = random_agent.step(legal_actions[ind_rand])
                    actions[ind_rand] = actions_rand
                if len(ind_ppo) > 0:
                    actions_ppo, log_probs, _, values, probs = self.agents[0].step(states[ind_ppo], legal_actions=legal_actions[ind_ppo])
                    actions[ind_ppo] = actions_ppo
                    log_probs_tot[ind_ppo] = log_probs
                    probs_tot[ind_ppo] = probs
                #actions, log_probs, _, values, probs = self.agents[0].step(states, legal_actions)
                rewards, dones, infos = self.vectorized_game.step(actions)
                round_num = i // num_players
                for game in range(len(player_ids)):
                    g = x * num_games + game
                    mb_states[g][round_num * num_players + player_ids[game]] = states[game].copy()
                    mb_actions[g][round_num * num_players + player_ids[game]] = actions[game]
                    #mb_values[g][round_num * num_players + player_ids[game]] = values[game][0]
                    mb_logprobs[g][round_num * num_players + player_ids[game]] = log_probs_tot[game]
                    mb_dones[g][round_num * num_players + player_ids[game]] = dones[game]
                    mb_rewards[g][round_num * num_players + player_ids[game]] = rewards[game]
                    mb_legal_actions[g][round_num * num_players + player_ids[game]] = legal_actions[game].copy()
                    mb_player_ids[g][round_num * num_players + player_ids[game]] = player_ids[game]
                    mb_probs[g][round_num * num_players + player_ids[game]] = probs_tot[game]
                if (i + 1) % num_players == 0:
                    for player in range(config.num_players):
                        _, perfect_states, _ = self.vectorized_game.get_perfect_state(player)
                        values = self.agents[0].step_value(perfect_states)
                        for game in range(len(perfect_states)):
                            mb_perfect_states[x * num_games + game][round_num * num_players + player] = perfect_states[game].copy()
                            mb_values[x*num_games + game][round_num * num_players + player] = values[game][0]
                player_ids, states, legal_actions = self.vectorized_game.get_state(None)
            reward = self.vectorized_game.get_payoffs()
            reward = reward.swapaxes(1, 0)
            for i in range(config.num_players):
                _, t_states, _= self.vectorized_game.get_perfect_state(i)
                t_values = self.agents[0].step_value(t_states)
                for g in range(len(t_values)):
                    mb_values[x * num_games + g][num_cards + i] = t_values[g]
            # reward = self.vectorized_game.get_payoffs()
            # reward = reward.swapaxes(1, 0)
            for g in range(num_games):
                mb_rewards[x * num_games + g][num_cards - config.num_players:num_cards] = reward[g]
        mb_states_2 = mb_states.swapaxes(1, 0)
        mb_perfect_states_2 = mb_perfect_states.swapaxes(1,0)
        mb_values_2 = mb_values.swapaxes(1,0)
        mb_logprobs_2 = mb_logprobs.swapaxes(1,0)
        mb_dones_2 = mb_dones.swapaxes(1,0)
        mb_rewards_2 = mb_rewards.swapaxes(1,0)
        mb_legal_actions_2 = mb_legal_actions.swapaxes(1,0)
        mb_actions_2 = mb_actions.swapaxes(1,0)
        mb_player_ids_2 = mb_player_ids.swapaxes(1,0)
        mb_probs_2 = mb_probs.swapaxes(1,0)
        mb_returns = np.zeros_like(mb_rewards_2)
        mb_advs = np.zeros_like(mb_rewards_2)
        for k in range(config.num_players):
            lastgaelam = 0
            for t in reversed(range(k, num_cards, config.num_players)):
                nextNonTerminal = mb_dones_2[t]
                nextValues = mb_values_2[t + config.num_players]
                delta = mb_rewards_2[t] + 0.99 * nextValues * nextNonTerminal - mb_values_2[t]
                mb_advs[t] = lastgaelam = delta + 0.99 * 0.99 * nextNonTerminal * lastgaelam
        mb_values_3 = mb_values_2[0:num_cards]
        mb_returns = mb_advs + mb_values_3
        statesr, perfect_statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, player_idsr, probsr = map(self.sf01,
                                                                                    (mb_states_2, mb_perfect_states_2,
                                                                                     mb_legal_actions_2,
                                                                                     mb_returns, mb_actions_2, mb_values_3,
                                                                                     mb_advs, mb_logprobs_2, mb_player_ids_2,
                                                                                     mb_probs_2))

        indices = np.where(player_idsr % 2 == 0)[0]
        statesr = statesr[indices]
        perfect_statesr = perfect_statesr[indices]
        legal_actionsr = legal_actionsr[indices]
        returnsr = returnsr[indices]
        actionsr = actionsr[indices]
        valuesr = valuesr[indices]
        advantagesr = advantagesr[indices]
        logprobsr = logprobsr[indices]
        probsr = probsr[indices]

        self.agents[0].run(statesr, perfect_statesr, legal_actionsr, returnsr, actionsr, valuesr, advantagesr, logprobsr, probsr)

    def run_ddpg(self):
        #replaybuffer = ReplayBuffer()
        random_agent = RandomAgent(config.num_cards)
        ddpg = self.agents[0]

        for x in range(1000000):
            player_id, state, legal_actions = self.game.start_game()
            for i in range(config.num_rounds):
                acs = [None] * config.num_players
                dones = []
                obs = [None] * config.num_players
                legals = [None] * config.num_players
                perfect_state = self.game.clone()
                for j in range(config.num_players):
                    obs[player_id] = torch.from_numpy(state.copy()).float()
                    legal_actions = torch.from_numpy(legal_actions.copy()).float()
                    action = ddpg.eval_step(torch.from_numpy(state).float(), legal_actions, j, explore=True)
                    acs[player_id] = action.detach()
                    legals[player_id] = legal_actions
                    action_int = torch.argmax(action, dim=-1).float().cpu().numpy()
                    reward, done, info = self.game.step(action_int)
                    player_id, state, legal_actions = self.game.get_state(None)
                    dones.append(done)
                perfect_next_state = self.game.clone()
                rews = self.game.get_payoffs_ddpg()
                ddpg.add(obs, perfect_state, legals, acs, rews, perfect_next_state, dones)

            if x > 1000:
                if x % 10 == 0:
                    ddpg.update(0, actor=True)
                else:
                    ddpg.update(0, actor=False)
                ddpg.update_all_targets()
            elif x >= 1000 and x % 10 == 0:
                ddpg.update(0, actor=True)
                ddpg.update_all_targets()
            if x >= 1040 and x % 20 == 0:
                print("Start simulation")
                res = self.simulate_ddpg_against_random(5000)
                print("Scores ", res)

    def build_complete_br(self, states_path, br_path):
        with open(states_path, 'rb') as file_handler:
            all_states = pickle.load(file_handler)
        random_agent = RandomAgent(config.num_cards)
        game = Manil2()
        br_agent1 = BestResponsePolicy(game, 0, random_agent)
        br_agent1.all_info_sets(all_states)
        br_agent2 = BestResponsePolicy(game, 1, random_agent)
        br_agent2.all_info_sets(all_states)
        br_agents = (br_agent1, br_agent2)
        with open(br_path, 'wb') as file_handler:
            pickle.dump(br_agents, file_handler)
            print("BR saved! Size: ", br_agent1.infosets.__len__() + br_agent2.infosets.__len__())


    def sf01(self, arr):
        """
        swap and then flatten axes 0 and 1
        """
        s = arr.shape
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

    def reorganize(self, trajectories, payoffs):
        ''' Reorganize the trajectory to make it RL friendly
        Args:
            trajectory (list): A list of trajectories
            payoffs (list): A list of payoffs for the players. Each entry corresponds to one player
        Returns:
            (list): A new trajectories that can be fed into RL algorithms.
        '''
        player_num = len(trajectories)
        new_trajectories = [[] for _ in range(player_num)]

        for player in range(player_num):
            for i in range(0, len(trajectories[player]) - 2, 2):
                if i == len(trajectories[player]) - 3:
                    reward = payoffs[player]
                    done = True
                else:
                    reward, done = 0, False
                transition = trajectories[player][i:i + 3].copy()
                transition.insert(2, reward)
                transition.append(done)

                new_trajectories[player].append(transition)
        return new_trajectories




