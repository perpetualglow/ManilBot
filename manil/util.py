from config import *
from copy import deepcopy
import config as config
import itertools
import pickle
import random
from manil.card import Card
import numpy as np

def index_to_card(index):
    y = index % num_rank_cards + min_rank
    x = index // num_rank_cards
    return Card(y, x)

def cards_to_matrix(hand):
    m = np.zeros((num_suits, num_rank_cards), dtype=int)
    for card in hand:
        if card is not None:
            x = card.suit
            y = card.rank - min_rank
            m[x][y] = 1
    return m

def card_to_list(hand=None, legal=None, current_trick_num=None, round_num=None, player=None, order=None, winner=None):
    m = np.zeros(16, dtype=int)
    ind = 0
    if hand == True:
        m[ind] = 1
    ind += 1
    if legal == True:
        m[ind] = 1
    ind += 1
    if current_trick_num is not None:
        m[ind + current_trick_num] = 1
    ind += config.num_players - 1
    if round_num is not None:
        m[ind + round_num] = 1
    ind += config.num_rounds - 1
    if player is not None:
        m[ind + player] = 1
    ind += config.num_players
    if order is not None:
        m[ind + order] = 1
    ind += config.num_players
    if winner is not None:
        m[ind] = 1
    return m

def card_to_list_perfect(hand=None, legal=None, current_trick_num=None, player=None):
    m = np.zeros(12, dtype=int)
    ind = 0
    if hand is not None:
        m[ind + hand] = 1
    ind += 1
    if legal == True:
        m[ind] = 1
    ind += 1
    if player is not None:
        m[ind + player] = 1
    ind += config.num_players
    if current_trick_num is not None:
        m[ind + current_trick_num] = 1
    ind += config.num_players - 1
    return m

def list_to_matrix(l):
    m = np.zeros((num_suits,num_rank_cards), dtype=int)
    for index in l:
        card = index_to_card(index)
        x = card.suit
        y = card.rank - min_rank
        m[x][y] = 1
    return m

def generate_game_states_2p(game, different_starters=True, states_path=None, num_states=None):
    deck = [Card(rank, suit).index for suit in range(config.num_suits)
            for rank in range(8 - config.num_rank_cards, 8)]
    deck = set(deck)
    all_deals = []
    first_player_cards = itertools.combinations(deck, config.player_cards_2p)
    for cards1 in first_player_cards:
        second_player_cards = deck - set(cards1)
        for cards2 in itertools.combinations(second_player_cards, config.player_cards_2p):
            up_cards = second_player_cards - set(cards2)
            for up_cards_1 in itertools.combinations(up_cards, config.up_cards_2p):
                up_cards_2 = up_cards - set(up_cards_1)
                for up_cards3 in itertools.combinations(up_cards_2, config.up_cards_2p):
                    down_cards_1 = up_cards_2 - set(up_cards3)
                    for down_cards_2 in itertools.combinations(down_cards_1, config.up_cards_2p):
                        down_cards3 = down_cards_1 - set(down_cards_2)
                        all_deals.append(list(cards1) + list(cards2) + list(down_cards_2) +  list(down_cards3) + list(up_cards_1) + list(up_cards3))
    random.shuffle(all_deals)
    print(str(len(all_deals)) + " deals generated!")
    all_states = []
    if num_states is None:
        for i, deal in enumerate(all_deals):
            state1 = game.new_initial_state(deal, 0)
            state2 = game.new_initial_state(deal, 1)
            all_states.append(deepcopy(state1))
            all_states.append(deepcopy(state2))
    else:
        for i, deal in enumerate(all_deals[0:num_states]):
            first_player = random.randint(0, 1)
            state1 = game.new_initial_state(deal, first_player)
            all_states.append(deepcopy(state1))
    print(str(len(all_states)) + " states generated!")
    if states_path is not None:
        with open(states_path, 'wb') as file_handler:
            pickle.dump(all_states, file_handler)
        print("States saved! Size: ", len(all_states))
        return all_states
    else:
        return all_states


def generate_game_states(game, different_starters=True, states_path=None):
    deck = [Card(rank, suit).index for suit in range(config.num_suits)
            for rank in range(8 - config.num_rank_cards, 8)]
    deck = set(deck)
    all_deals = []
    if game.num_players == 2:
        first_player_cards = itertools.combinations(deck, config.num_player_cards)
        for cards1 in first_player_cards:
            cards2 = deck - set(cards1)
            all_deals.append([list(cards1),list(cards2)])
    else:
        first_player_cards = itertools.combinations(deck, config.num_player_cards)
        for cards1 in first_player_cards:
            second_player_cards = deck - set(cards1)
            for cards2 in itertools.combinations(second_player_cards, config.num_player_cards):
                third_player_cards = second_player_cards - set(cards2)
                for cards3 in itertools.combinations(third_player_cards, config.num_player_cards):
                    cards4 = tuple(third_player_cards - set(cards3))
                    all_deals.append([list(cards1), list(cards2), list(cards3), list(cards4)])
    all_states = []
    ind = 0
    for i, deal in enumerate(all_deals):
        if game.num_players == 2:
            if different_starters:
                state1 = game.new_initial_state(deal, 0, ind)
                all_states.append(deepcopy(state1))
                state2 = game.new_initial_state(deal, 1, ind + 1)
                all_states.append(deepcopy(state2))
                ind += 2
            else:
                state1 = game.new_initial_state(deal, 0, ind)
                all_states.append(deepcopy(state1))
                ind += 1
        elif game.num_players == 4:
            if different_starters:
                state1 = game.new_initial_state(deal, 0, ind)
                all_states.append(deepcopy(state1))
                state2 = game.new_initial_state(deal, 1, ind + 1)
                all_states.append(deepcopy(state2))
                state3 = game.new_initial_state(deal, 2, ind + 2)
                all_states.append(deepcopy(state3))
                state4 = game.new_initial_state(deal, 3, ind + 3)
                all_states.append(deepcopy(state4))
                ind += 4
            else:
                state1 = game.new_initial_state(deal, 0, ind)
                all_states.append(deepcopy(state1))
                ind += 1
    print(str(len(all_states)) + " states generated!")
    if states_path is not None:
        with open(states_path, 'wb') as file_handler:
            pickle.dump(all_states, file_handler)
            print("States saved! Size: ", len(all_states))
    else:
        return all_states



