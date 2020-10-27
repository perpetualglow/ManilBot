import config as config

class Rules:

    def __init__(self, mull, trump):
        self.mull = mull
        self.trump = trump

    def judge(self, plays):
        if self.mull:
            return self.judge_mull(plays)
        else:
            return self.judge_trump(plays)

    def judge_mull(self, plays):
        if len(plays) > 0:
            best = plays[0]
            for p in plays:
                x = p.card
                if (x.suit == best.card.suit and x > best.card):
                    best = p
            points = 0
            for p in plays:
                points += p.card.get_value()
            return best.player, points, False
        else:
            return None

    def judge_trump(self, plays):
        if len(plays) > 0:
            best = plays[0]
            for p in plays:
                x = p.card
                if best.card.suit == self.trump:
                    if x.suit == best.card.suit and x > best.card:
                        best = p
                else:
                    if (p.card.suit == best.card.suit and p.card > best.card) or p.card.suit == self.trump:
                        best = p
            points = 0
            for p in plays:
                points += p.card.get_value()
            return best.player, points, best.card.suit == self.trump
        else:
            return None


    def get_legal_actions(self, player, hand, plays):
        if self.mull:
            return self.get_legal_actions_mull(player, hand, plays)
        else:
            return self.get_legal_actions_trump(player, hand, plays)


    def get_legal_actions_mull(self, player, hand, plays):
        if config.num_players == 2:
            return self.get_legal_actions_mull_2(hand, plays)
        elif config.num_players == 4:
            return self.get_legal_actions_mull_4(player, hand, plays)

    def get_legal_actions_trump(self, player, hand, plays):
        if config.num_players == 2:
            return self.get_legal_actions_trump_2(hand, plays)
        elif config.num_players == 4:
            return self.get_legal_actions_trump_4(player, hand, plays)
        return

    def get_legal_actions_mull_2(self, hand, plays):
        res = []
        if len(plays) == 0:
            for c in hand:
                res.append(c.index)
            return res
        else:
            suit_cards = list(filter(lambda x: (x.suit == plays[0].card.suit), hand))
            if len(suit_cards) > 0:
                higher_cards = list(filter(lambda x: (x > plays[0].card), suit_cards))
                if len(higher_cards) > 0:
                    for c in higher_cards:
                        res.append(c.index)
                    return res
                else:
                    for c in suit_cards:
                        res.append(c.index)
                    return res
            else:
                for c in hand:
                    res.append(c.index)
                return res

    def get_legal_actions_trump_2(self, hand, plays):
        res = []
        if len(plays) == 0:
            for card in hand:
                res.append(card.index)
            return res
        else:
            suit_cards = list(filter(lambda x: (x.suit == plays[0].card.suit), hand))
            if len(suit_cards) > 0:
                higher_cards = list(filter(lambda x: (x > plays[0].card), suit_cards))
                if len(higher_cards) > 0:
                    for c in higher_cards:
                        res.append(c.index)
                    return res
                else:
                    for c in suit_cards:
                        res.append(c.index)
                    return res
            else:
                if plays[0].card.suit != self.trump:
                    trump_cards = list(filter(lambda x: (x.suit == self.trump), hand))
                    if len(trump_cards) > 0:
                        for c in trump_cards:
                            res.append(c.index)
                        return res
                    else:
                        for c in hand:
                            res.append(c.index)
                        return res
                else:
                    for c in hand:
                        res.append(c.index)
                    return res

    def get_legal_actions_trump_4(self, player, hand, plays):
        if len(plays) == 0:
            return self.ret(hand)
        elif len(plays) == 1:
            return self.get_legal_actions_trump_2(hand, [plays[0]])
        elif len(plays) == 2:
            winner, _, bought = self.judge_trump(plays)
            ind = 0
            for i, p in enumerate(plays):
                if winner == p.player:
                    ind = i
            if plays[ind].card.suit != self.trump:
                if ind == 0:
                    suit_cards = list(filter(lambda c: c.suit == plays[0].card.suit, hand))
                    if len(suit_cards) > 0:
                        return self.ret(suit_cards)
                    else:
                        return self.ret(hand)
                else:
                    suit_cards = list(filter(lambda c: c.suit == plays[0].card.suit, hand))
                    if len(suit_cards) > 0:
                        higher_suit_cards = list(filter(lambda c: c > plays[1].card, suit_cards))
                        if len(higher_suit_cards) > 0:
                            return self.ret(higher_suit_cards)
                        else:
                            return self.ret(suit_cards)
                    else:
                        trump_cards = list(filter(lambda c: c.suit == self.trump, hand))
                        if len(trump_cards) > 0:
                            return self.ret(trump_cards)
                        else:
                            return self.ret(hand)
            if plays[0].card.suit == self.trump:
                trump_cards = list(filter(lambda c: c.suit == self.trump, hand))
                if len(trump_cards) > 0:
                    if ind == 1:
                        higher_trump_cards = list(filter(lambda c: c > plays[ind].card, trump_cards))
                        if len(higher_trump_cards) > 0:
                            return self.ret(higher_trump_cards)
                        else:
                            return self.ret(trump_cards)
                    else:
                        return self.ret(trump_cards)
                else:
                    return self.ret(hand)
            if plays[1].card.suit == self.trump:
                suit_cards = list(filter(lambda c: c.suit == plays[0].card.suit, hand))
                if len(suit_cards) > 0:
                    return self.ret(suit_cards)
                else:
                    higher_trump_cards = list(filter(lambda c: c.suit == self.trump and c > plays[1].card, hand))
                    if len(higher_trump_cards) > 0:
                        return self.ret(higher_trump_cards)
                    else:
                        other_cards = list(filter(lambda c: c.suit != self.trump, hand))
                        if len(other_cards) > 0:
                            return self.ret(other_cards)
                        else:
                            return self.ret(hand)
        elif len(plays) == 3:
            winner, _, bought = self.judge_trump(plays)
            ind = 0
            for i, p in enumerate(plays):
                if winner == p.player:
                    ind = i
            if plays[ind].card.suit != self.trump:
                if ind == 0 or ind == 2:
                    suit_cards = list(filter(lambda c: c.suit == plays[0].card.suit, hand))
                    if len(suit_cards) > 0:
                        higher_suit_cards = list(filter(lambda c: c > plays[ind].card, suit_cards))
                        if len(higher_suit_cards) > 0:
                            return self.ret(higher_suit_cards)
                        else:
                            return self.ret(suit_cards)
                    else:
                        trump_cards = list(filter(lambda c: c.suit == self.trump, hand))
                        if len(trump_cards) > 0:
                            return self.ret(trump_cards)
                        else:
                            return self.ret(hand)
                else:
                    suit_cards = list(filter(lambda c: c.suit == plays[0].card.suit, hand))
                    if len(suit_cards) > 0:
                        return self.ret(suit_cards)
                    else:
                        return self.ret(hand)
            if plays[0].card.suit == self.trump:
                suit_cards = list(filter(lambda c: c.suit == self.trump, hand))
                if len(suit_cards) > 0:
                    if ind == 0 or ind == 2:
                        higher_cards = list(filter(lambda c: c > plays[ind].card, suit_cards))
                        if len(higher_cards) > 0:
                            return self.ret(higher_cards)
                        else:
                            return self.ret(suit_cards)
                    else:
                        return self.ret(suit_cards)
                else:
                    return self.ret(hand)
            if plays[ind].card.suit == self.trump:
                if ind == 0 or ind == 2:
                    suit_cards = list(filter(lambda c: c.suit == plays[0].card.suit, hand))
                    if len(suit_cards) > 0:
                        return self.ret(suit_cards)
                    else:
                        higher_trump_cards = list(filter(lambda c: c.suit == self.trump and c > plays[ind].card, hand))
                        if len(higher_trump_cards) > 0:
                            return self.ret(higher_trump_cards)
                        else:
                            other_cards = list(filter(lambda c: c.suit != self.trump, hand))
                            if len(other_cards) > 0:
                                return self.ret(other_cards)
                            else:
                                return self.ret(hand)
                else:
                    suit_cards = list(filter(lambda c: c.suit == plays[0].card.suit, hand))
                    if len(suit_cards) > 0:
                        return self.ret(suit_cards)
                    else:
                        return self.ret(hand)

    def ret(self, cards):
        res = []
        for c in cards:
            res.append(c.index)
        return res

    def get_legal_actions_mull_4(self, player, hand, plays):
        res = []
        if len(plays) == 0:
            for card in hand:
                res.append(card.index)
            return res
        else:
            winner, _ , _ = self.judge_mull(plays)
            table_card = None
            for play in plays:
                if play.player == winner:
                    table_card = play.card
            if (winner + 2) % config.num_players == player:
                suit_cards = list(filter(lambda x: (x.suit == table_card.suit), hand))
                if len(suit_cards) > 0:
                    for c in suit_cards:
                        res.append(c.index)
                    return res
                else:
                    for c in hand:
                        res.append(c.index)
                    return res
            else:
                higher_cards = list(filter(lambda x: (x.suit == table_card.suit and x > table_card), hand))
                if len(higher_cards) > 0:
                    for c in higher_cards:
                        res.append(c.index)
                    return res
                else:
                    suit_cards = list(filter(lambda x: (x.suit == table_card.suit), hand))
                    if len(suit_cards) > 0:
                        for c in suit_cards:
                            res.append(c.index)
                        return res
                    else:
                        for c in hand:
                            res.append(c.index)
                        return res