import config as c

class Card:

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.index = (c.num_cards // c.num_suits) * self.suit + (self.rank - c.min_rank)

    def __repr__(self):
        letter = c.LETTERS.get(self.rank)
        return "<%s %s>" % (letter, c.SUITS_TEXT.get(self.suit))

    def get_index(self):
        return (c.num_cards // c.num_suits) * self.suit + self.rank - c.min_rank

    def get_value(self):
        return c.VALUES.get(self.rank)

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit

    def __lt__(self, other):
        return self.rank < other.rank