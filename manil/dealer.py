from random import shuffle
from copy import copy
import itertools

from manil.card import Card
import config as c

class Dealer():

    def __init__(self):
        self.deck = [Card(rank, suit) for suit in range(c.num_suits)
                     for rank in range(8 - c.num_rank_cards, 8)]
        shuffle(self.deck)

    def reset(self):
        self.deck = [Card(rank, suit) for suit in range(c.num_suits)
                     for rank in range(8 - c.num_rank_cards, 8)]
        shuffle(self.deck)

    def all_combinations(self):
        deck = [Card(rank, suit) for suit in range(c.num_suits)
                     for rank in range(8 - c.num_rank_cards, 8)]
        n_cards = c.num_cards // c.num_players
        poss_cards = list(itertools.combinations(deck, n_cards))
        return list(itertools.combinations(poss_cards, c.num_players))

    def deal_cards(self, players):
        i = 0
        n_cards = c.num_cards // c.num_players
        for p in players:
            p.deal_hand(self.deck[i:i + n_cards])
            i += n_cards

    def utility(self):
        util = 0
        for card in self.deck:
            util += card.get_value()
        return util
