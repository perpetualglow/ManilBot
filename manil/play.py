

class Play:
    def __init__(self, player, card, round_num, round_card_num):
        self.player = player
        self.card = card
        self.round_num = round_num
        self.round_card_num = round_card_num

    def __repr__(self):
        return "%s" % (self.card)