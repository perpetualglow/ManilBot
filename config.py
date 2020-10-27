import torch

SUITS_TEXT = {0: 'spades', 1: 'hearts', 2: 'diamonds', 3: 'clubs'}
LETTERS = {0: '7', 1: '8', 2: '9', 3: 'J', 4:'Q', 5:'K', 6:'A', 7: '10'}
VALUES= {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}

utility_sum = 48

num_players = 2
num_suits = 2
num_cards = 8
num_player_cards = num_cards // num_players
num_rank_cards = num_cards // num_suits
num_suit_cards = num_cards // num_rank_cards
num_rounds = num_player_cards
min_rank = 8 - num_rank_cards
mull = True

player_cards_2p = 2
down_cards_2p = 1
up_cards_2p = 1

state_shape = [2 * num_players + num_rounds + 2 + 3, num_suits, num_rank_cards]
perfect_state_shape = [5, num_suits, num_rank_cards]
#perfect_state_shape = [num_cards, 2 + 2 * num_players + num_rounds - 1 + num_players - 1]
#perfect_state_shape = [num_cards + 1, 12]
state_shape_2p = [7 + 2*num_players + num_rounds, num_suits, num_rank_cards]
state_shape_2p_2 = (7 + 2*num_players + num_rounds) * num_suits * num_rank_cards
perfect_information = False

NUM_CONCURRENT_GAMES = 64
NUM_TOTAL_GAMES = 64

EVAL_NUM = 500

device = torch.device('cpu')
learning_rate = 0.001
epsilon = 0.0001
discount_rate = 0.995
entropy_coefficient = 0.01
ppo_clip = 0.3
gradient_clip = 0.5
num_actions = 32
network_shape = [256]
