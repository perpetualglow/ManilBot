from random import randint, choice
import numpy as np
from copy import deepcopy, copy
from multiprocessing import Pipe, Process

from manil.dealer import Dealer
from manil.player import Player
from manil.rules import Rules
from manil.play import Play
from manil.util import *
import config as c

class Manil:

    def __init__(self, eval=True):
        self.trump = None
        self.players = None
        self.dealer = Dealer()
        self.legal_actions = None
        self.utility = self.dealer.utility()
        self.history = []
        self.num_players = c.num_players
        self.state_shape = tuple(c.state_shape)
        self.perfect_state_shape = tuple(c.perfect_state_shape)
        self.perfect = c.perfect_information
        self.eval = eval

    def start_game(self):
        self.history = []
        self.round_num = 0
        self.moves = []
        self.current_trick = []
        self.players = [Player(i) for i in range(self.num_players)]
        self.dealer.reset()
        self.utility = self.dealer.utility()
        self.dealer.deal_cards(self.players)
        self.current_player = randint(0, self.num_players - 1)
        self.first_player = self.current_player
        self.winners = []
        #self.current_player = 0
        if not c.mull:
            #self.trump = randint(0, c.num_suits - 1)
            self.trump = 0
            self.rules = Rules(False, self.trump)
        else:
            self.trump = None
            self.rules = Rules(True, self.trump)
        self.legal_actions = self.calc_legal_actions()
        return self.current_player, self.encode_state(self.current_player), self.encode_legal_actions(self.legal_actions)

    def start_game2(self):
        self.history = []
        self.round_num = 0
        self.moves = []
        self.current_trick = []
        self.players = [Player(i) for i in range(self.num_players)]
        self.dealer.reset()
        self.utility = self.dealer.utility()
        self.dealer.deal_cards(self.players)
        self.current_player = randint(0, self.num_players - 1)
        self.first_player = self.current_player
        self.winners = []
        #self.current_player = 0
        if not c.mull:
            #self.trump = randint(0, c.num_suits - 1)
            self.trump = 0
            self.rules = Rules(False, self.trump)
        else:
            self.trump = None
            self.rules = Rules(True, self.trump)
        self.legal_actions = self.calc_legal_actions()
        return self.current_player, self.encode_state(self.current_player), self.encode_legal_actions2(self.legal_actions)

    def step(self, action):
        ind = self.decode_action(action)
        card = self.players[self.current_player].play_card(ind)
        play = Play(self.current_player, card, self.round_num, len(self.current_trick))
        self.current_trick.append(play)
        self.moves.append(play)
        self.current_player = (self.current_player + 1) % self.num_players
        if len(self.current_trick) == self.num_players:
            self.finish_trick()
        if len(self.players[self.current_player].hand) == 0:
            self.finish_game()
        payoffs = 0
        done = 1
        if len(self.moves) > c.num_cards - 2:
            done = 0
        self.legal_actions = self.calc_legal_actions()
        return payoffs, done, self.round_num

    def step2(self, action):
        card = index_to_card(action)
        wrong = False
        if card in self.players[self.current_player].hand:
            ind = self.players[self.current_player].hand.index(card)
            played_action = action
        else:
            wrong = True
            if not self.eval:
                self.players[self.current_player].points -= 100
            played_action = choice(self.legal_actions)
            card = index_to_card(played_action)
            ind = self.players[self.current_player].hand.index(card)
        card = self.players[self.current_player].play_card(ind)
        play = Play(self.current_player, card, self.round_num, len(self.current_trick))
        self.current_trick.append(play)
        self.moves.append(play)
        self.current_player = (self.current_player + 1) % self.num_players
        if len(self.current_trick) == self.num_players:
            self.finish_trick()
        if len(self.players[self.current_player].hand) == 0:
            self.finish_game()
        payoffs = 0
        done = 1
        if len(self.moves) > c.num_cards - self.num_players:
            done = 0
        self.legal_actions = self.calc_legal_actions()
        return played_action, payoffs, done, wrong

    def step_back(self):
        if len(self.history) > 0:
            self.current_player, self.round_num, self.players, self.current_trick, self.moves, self.legal_actions = self.history.pop()

    def get_state(self, id):
        if id is None:
            return self.current_player, self.encode_state(self.current_player), self.encode_legal_actions(
                self.legal_actions)
        return id, self.encode_state(id), self.encode_legal_actions(self.legal_actions)

    def get_perfect_state(self, id=None):
        if id is None:
            id = self.current_player
        return id, self.encode_state_perfect(id), None

    def get_state2(self, id):
        if id is None:
            return self.current_player, self.encode_state(self.current_player), self.encode_legal_actions2(
                self.legal_actions)
        return id, self.encode_state(id), self.encode_legal_actions2(self.legal_actions)

    def get_endstate(self, id):
        if self.perfect:
            return self.get_state(id)
        obs = np.zeros(self.state_shape, dtype=float)
        order_cards = []
        player_cards = []
        round_cards = []
        winners = []
        for winner in self.winners:
            winners.append((winner - id) % self.num_players)
        for i in range(self.num_players):
            order_cards.append([])
            player_cards.append([])
        n_rounds = (c.num_cards // self.num_players)
        for i in range(n_rounds):
            round_cards.append([])
        for m in self.moves:
            player_ind = (m.player - id) % self.num_players
            order_ind = m.round_card_num
            round_cards[m.round_num].append(m.card)
            player_cards[player_ind].append(m.card)
            order_cards[order_ind].append(m.card)
        ind = 2
        for i in range(self.num_players):
            obs[ind + i] = cards_to_matrix(player_cards[i])
        ind += self.num_players
        for i in range(self.num_players):
            obs[ind + i] = cards_to_matrix(order_cards[i])
        ind += self.num_players
        for i in range(n_rounds):
            obs[ind + i] = cards_to_matrix(round_cards[i])
        return id, obs, None

    def finish_trick(self):
        winner, points, _ = self.rules.judge(self.current_trick)
        self.players[winner].points += points
        self.players[winner].reward = points
        self.players[(winner + 2) % self.num_players].reward = points
        self.players[(winner + 1) % self.num_players].reward = -points
        self.players[(winner + 3) % self.num_players].reward = -points
        if self.num_players > 2:
             self.players[(winner + 2) % self.num_players].points += points
        self.current_player = winner
        # self.current_player = (self.current_player + 1) % self.num_players
        for move in self.current_trick:
            if move.player == winner:
                self.winners.append(move.card)
        self.current_trick = []
        self.round_num += 1
        return

    def finish_game(self):
        return

    def get_payoffs_ddpg(self):
        if len(self.moves) != num_cards:
            return [0,0,0,0]
        else:
            payoffs = []
            for player in self.players:
                # payoffs.append((player.points - self.utility/2)/(self.utility/2))
                payoffs.append(player.points - self.utility/2)
            return payoffs

    def get_ddpg_rewards(self):
        rewards = []
        for player in self.players:
            rewards.append(player.points)
        return rewards

    def get_payoffs(self):
        payoffs = []
        for player in self.players:
            payoffs.append(player.points)
        return payoffs

    def clone(self):
        return deepcopy(self)

    def get_payoffs_ppo(self):
        payoffs = []
        for player in self.players:
            payoffs.append(float(player.points - self.utility/2) / (self.utility / 2))
        return payoffs

    def get_payoffs_br(self):
        payoffs = []
        payoffs.append(self.players[0].points - 7.5)
        payoffs.append(self.players[1].points - 7.5)
        return payoffs

    def get_current_player_id(self):
        return self.current_player

    def get_current_player_expl(self):
        return self.current_player % 2

    def is_over(self):
        return len(self.moves) == c.num_cards

    def get_legal_actions(self):
        return self.legal_actions

    def calc_legal_actions(self):
        return self.rules.get_legal_actions(self.current_player, self.players[self.current_player].hand, self.current_trick)

    def encode_legal_actions(self, legal_actions):
        legal = np.ones(c.num_cards) * -1e16
        legal[legal_actions] = 0
        return legal

    def encode_legal_actions2(self, legal_actions):
        legal = np.zeros(c.num_cards)
        legal[legal_actions] = 1
        return legal

    def encode_state(self, id=None):
        if id is None:
            id = self.current_player
        if self.perfect == True:
            return self.encode_state_perfect(id)
        else:
            return self.encode_state_imp(id)

    def encode_state_perfect(self, id=None):
        if id is None:
            id = self.current_player
        obs = np.zeros(self.perfect_state_shape, dtype=float)
        ind = 0
        for i in range(self.num_players):
            player_ind = (id + i) % self.num_players
            obs[ind + i] = cards_to_matrix(self.players[player_ind].hand)
        ind += self.num_players
        obs = obs.flatten()
        score = self.players[id].points
        obs[16*ind + 1] = score
        return obs.flatten()

    def encode_state_imp(self, id):
        obs = np.zeros(self.state_shape, dtype=float)
        hand = self.players[id].hand
        order_cards = []
        player_cards = []
        round_cards = []
        for i in range(self.num_players):
            order_cards.append([])
            player_cards.append([])
        n_rounds = (c.num_cards // self.num_players)
        for i in range(n_rounds):
            round_cards.append([])
        for m in self.moves:
            if m.round_num < self.round_num:
                player_ind = (m.player - id) % self.num_players
                order_ind = m.round_card_num
                round_cards[m.round_num].append(m.card)
                player_cards[player_ind].append(m.card)
                order_cards[order_ind].append(m.card)
        obs[0] = cards_to_matrix(hand)
        obs[1] = list_to_matrix(self.legal_actions)
        ind = 2
        for i in range(self.num_players):
            obs[ind + i] = cards_to_matrix(player_cards[i])
        ind += self.num_players
        for i in range(self.num_players):
            obs[ind + i] = cards_to_matrix(order_cards[i])
        ind += self.num_players
        for i in range(n_rounds):
            obs[ind + i] = cards_to_matrix(round_cards[i])
        ind += n_rounds - 1
        for i in range(len(self.current_trick)):
            obs[ind + i] = cards_to_matrix([self.current_trick[i].card])
        ind += self.num_players - 1
        '''
        for i in range(len(winners)):
            obs[ind + i][winners[i]][0] = 1
        ind += n_rounds
        '''
        return obs.flatten()

    def decode_action(self, action):
        card = index_to_card(action)
        if card in self.players[self.current_player].hand:
            return self.players[self.current_player].hand.index(card)
        else:
            print('WRONG ACTION')
            return None

    def get_state_shape(self):
        return self.state_shape

    def get_perfect_state_shape(self):
        return self.perfect_state_shape

def worker(remote, parent_remote):
    parent_remote.close()
    game = Manil()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            reward, done, info = game.step(data)
            remote.send((reward, done, info))
        elif cmd == 'reset':
            player_id, state, legal_actions = game.start_game()
            remote.send((player_id, state, legal_actions))
        elif cmd == 'reset2':
            player_id, state, legal_actions = game.start_game2()
            remote.send((player_id, state.flatten(), legal_actions))
        elif cmd == 'getCurrState':
            player_id, state, legal_actions = game.get_state(data)
            remote.send((player_id, state, legal_actions))
        elif cmd == 'getCurrState2':
            player_id, state, legal_actions = game.get_state2(data)
            remote.send((player_id, state.flatten(), legal_actions))
        elif cmd == 'getPayoffs':
            payoffs = game.get_payoffs()
            remote.send(payoffs)
        elif cmd == 'getPerfectState':
            player_id, state, legal_actions = game.get_perfect_state(data)
            remote.send((player_id, state, legal_actions))
        elif cmd == 'getEndState':
            player_id, state, legal_actions = game.get_endstate(data)
            remote.send((player_id, state, legal_actions))
        elif cmd == 'getEndState2':
            player_id, state, legal_actions = game.get_endstate2(data)
            remote.send((player_id, state, legal_actions))
        elif cmd == 'step2':
            action, reward, done, info = game.step2(data)
            remote.send((action, reward, done, info))
        elif cmd == 'close':
            remote.close()
            break
        else:
            print("Invalid command sent by remote")
            break


class VectorizedManil(object):
    def __init__(self, nGames):

        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nGames)])
        self.ps = [Process(target=worker, args=(work_remote, remote)) for (work_remote, remote) in
                   zip(self.work_remotes, self.remotes)]

        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        rewards, dones, infos = zip(*results)
        return rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step2_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step2', action))
        self.waiting = True

    def step2_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        actions, rewards, dones, infos = zip(*results)
        return actions, rewards, dones, infos

    def step2(self, actions):
        self.step2_async(actions)
        return self.step2_wait()

    def currStates_async(self, id):
        for remote in self.remotes:
            remote.send(('getCurrState', id))
        self.waiting = True

    def currStates_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        player_id, state, legal_actions = zip(*results)
        return np.stack(player_id), np.stack(state), np.stack(legal_actions)

    def get_state(self, id):
        self.currStates_async(id)
        return self.currStates_wait()

    def currStates2_async(self, id):
        for remote in self.remotes:
            remote.send(('getCurrState2', id))
        self.waiting = True

    def currStates2_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        player_id, state, legal_actions = zip(*results)
        return np.stack(player_id), np.stack(state), np.stack(legal_actions)

    def get_state2(self, id):
        self.currStates2_async(id)
        return self.currStates2_wait()

    def perfect_state_async(self, id):
        for remote in self.remotes:
            remote.send(('getPerfectState', id))
        self.waiting = True

    def perfect_state_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        player_id, state, legal_actions = zip(*results)
        return np.stack(player_id), np.stack(state), np.stack(legal_actions)

    def get_perfect_state(self, id=None):
        self.perfect_state_async(id)
        return self.perfect_state_wait()

    def endstate_async(self, id):
        for remote in self.remotes:
            remote.send(('getEndState', id))
        self.waiting = True

    def endstate_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        player_id, state, legal_actions = zip(*results)
        return np.stack(player_id), np.stack(state), np.stack(legal_actions)

    def get_endstate(self, id):
        self.endstate_async(id)
        return self.endstate_wait()

    def reset_async(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        self.waiting = True

    def reset_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        player_id, state, legal_actions = zip(*results)
        return np.stack(player_id), np.stack(state), np.stack(legal_actions)

    def reset(self):
        self.reset_async()
        return self.reset_wait()

    def reset2_async(self):
        for remote in self.remotes:
            remote.send(('reset2', None))
        self.waiting = True

    def reset2_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        player_id, state, legal_actions = zip(*results)
        return np.stack(player_id), np.stack(state), np.stack(legal_actions)

    def reset2(self):
        self.reset2_async()
        return self.reset2_wait()

    def get_payoffs(self):
        self.payoffs_async()
        return self.payoffs_wait()

    def payoffs_async(self):
        for remote in self.remotes:
            remote.send(('getPayoffs', None))
        self.waiting = True

    def payoffs_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        payoffs = zip(*results)
        return np.stack(payoffs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True





