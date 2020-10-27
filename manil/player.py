from copy import copy
from manil.util import *

class Player:

    def __init__(self, id):
        self._id = id
        self.hand = []
        self.points = 0
        self.played_cards = []
        self.original_hand = []
        self.up_cards = []
        self.down_cards = []
        self.reward = 0

    def get_id(self):
        return self._id

    def deal_hand(self, hand):
        self.hand = copy(hand)
        self.original_hand = copy(hand)
        self.hand.sort(key=lambda x: x.get_index())

    def set_hand(self, indexes):
        for ind in indexes:
            self.hand.append(index_to_card(ind))
            self.original_hand.append(index_to_card(ind))

    def play_card(self, index):
        c = self.hand.pop(index)
        self.played_cards.append(c)
        return c


