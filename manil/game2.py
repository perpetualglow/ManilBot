from random import randint
import numpy as np
from copy import deepcopy, copy
from multiprocessing import Pipe, Process

from manil.dealer import Dealer
from manil.player import Player
from manil.rules import Rules
from manil.play import Play
from manil.util import *
import config as c

class Manil2:

    def __init__(self, record = False):
        self.trump = None
        self.players = None
        self.dealer = Dealer()
        self.dealer.reset()
        self.utility = self.dealer.utility()
        self.legal_actions = None
        self.history = []
        self.num_players = c.num_players
        self.perfect = False
        self.state_shape = tuple(c.state_shape)
        self.perfect_state_shape = tuple(c.perfect_state_shape)
        self.winners = []
        self.index = 0

    def new_initial_state(self, deal=None, current_player=None, index=0):
        self.index = index
        self.round_num = 0
        self.moves = []
        self.current_trick = []
        self.winners = []
        self.players = [Player(i) for i in range(self.num_players)]
        if deal == None:
            self.dealer.reset()
            self.dealer.deal_cards(self.players)
        else:
            for i, cards in enumerate(deal):
                self.players[i].set_hand(cards)
        if current_player == None:
            self.current_player = randint(0, self.num_players - 1)
        else:
            self.current_player = current_player
        #self.current_player = 0
        if not c.mull:
            self.trump = randint(0, c.num_suits - 1)
            self.rules = Rules(c.mull, self.trump)
        else:
            self.trump = None
            self.rules = Rules(c.mull, self.trump)
        self.legal_actions = self.calc_legal_actions()
        return self


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
        self.legal_actions = self.calc_legal_actions()

    def get_state(self, id):
        if id is None:
            return self.current_player, self.encode_state(self.current_player), self.encode_legal_actions(
                self.legal_actions)
        return id, self.encode_state(id), self.encode_legal_actions(self.legal_actions)

    def information_state_string(self, id=None):
        if id == None:
            return self.encode_state(self.current_player).tostring()
        return self.encode_state(id).tostring()

    def clone(self):
        return deepcopy(self)

    def child(self, action):
        clone = deepcopy(self)
        clone.step(action)
        return clone

    def finish_trick(self):
        winner, points, _ = self.rules.judge(self.current_trick)
        self.players[winner].points += points
        if self.num_players > 2:
            self.players[(winner + 2) % self.num_players].points += points
        self.current_player = winner
        self.winners.append(winner)
        self.current_trick = []
        self.round_num += 1
        return

    def finish_game(self):
        return

    def num_players(self):
        return 2

    def get_payoffs(self):
        payoffs = []
        for player in self.players:
            payoffs.append(player.points)
        return payoffs

    def get_payoffs_br(self):
        payoffs = []
        payoffs.append(self.players[0].points - 7.5)
        payoffs.append(self.players[1].points - 7.5)
        return payoffs

    def player_return(self, id):
        return self.players[id].points

    def utility_sum(self):
        return self.utility

    def get_current_player_id(self):
        return self.current_player

    def get_current_player(self):
        return self.current_player % 2

    def is_over(self):
        return len(self.moves) == c.num_cards

    def get_legal_actions(self):
        return self.legal_actions

    def calc_legal_actions(self):
        return self.rules.get_legal_actions(self.current_player, self.players[self.current_player].hand, self.current_trick)

    def legal_actions_mask(self):
        legal = [0] * c.num_cards
        for act in self.calc_legal_actions():
            legal[act] = 1
        return legal

    def encode_legal_actions(self, legal_actions):
        legal = np.ones(c.num_cards) * -1e16
        legal[legal_actions] = 0
        return legal

    def encode_state(self, id=None):
        if id is None:
            id = self.current_player
        if self.perfect == True:
            return self.encode_state_perfect(id)
        else:
            return self.encode_state_imp(id)

    def encode_state_perfect(self, id):
        obs = np.zeros(self.perfect_state_shape, dtype=float)
        obs[0] = list_to_matrix(self.legal_actions)
        played_cards = []
        for m in self.moves:
            played_cards.append(m.card)
        obs[1] = cards_to_matrix(played_cards)
        for i in range(self.num_players):
            player_ind = (id + i) % self.num_players
            obs[2 + i] = cards_to_matrix(self.players[player_ind].hand)
        for i in range(len(self.current_trick)):
            obs[2 + self.num_players + i] = cards_to_matrix([self.current_trick[i].card])
        return obs

    def encode_state_imp(self, id):
        obs = np.zeros(self.state_shape, dtype=float)
        hand = self.players[id].hand
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
        #for i in range(len(self.current_trick)):
         #   obs[ind + i] = cards_to_matrix([self.current_trick[i].card])
        #ind += self.num_players - 1
        '''
        for i in range(len(winners)):
            obs[ind + i][winners[i]][0] = 1
        ind += n_rounds
        '''
        return obs

    def decode_action(self, action):
        card = index_to_card(action)
        if card in self.players[self.current_player].hand:
            return self.players[self.current_player].hand.index(card)
        else:
            print('WRONG ACTION')
            return None





