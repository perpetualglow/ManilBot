from random import randint, choice, shuffle
import numpy as np
from copy import deepcopy, copy
from multiprocessing import Pipe, Process
import collections

from manil.dealer import Dealer
from manil.player import Player
from manil.rules import Rules
from manil.play import Play
from manil.util import *
import config as c

class Manil_2p:

    def __init__(self):
        self.trump = None
        self.players = None
        self.dealer = Dealer()
        self.legal_actions = None
        self.dealer.reset()
        self.utility = self.dealer.utility()
        self.history = []
        self.num_players = 2
        self.state_shape = tuple(c.state_shape_2p)
        self.perfect_state_shape = tuple(c.perfect_state_shape)
        self.perfect = c.perfect_information
        self.down_cards = [[], []]
        self.up_cards = [[], []]
        self.played_up_cards = [[],[]]
        self.discounts = [1,1]

    def start_game(self, deal=None, current_player=None):
        self.over = False
        self.round_num = 0
        self.num_rounds = config.num_rounds
        self.played_up_cards = [[], []]
        self.moves = []
        self.current_trick = []
        self.players = [Player(i) for i in range(self.num_players)]
        self.deal_cards(deal)
        if current_player is None:
            self.current_player = randint(0, self.num_players - 1)
        else:
            self.current_player = current_player
        self.first_player = self.current_player
        self.to_open = []
        if not c.mull:
            #self.trump = randint(0, c.num_suits - 1)
            self.trump = 0
            self.rules = Rules(False, self.trump)
        else:
            self.trump = None
            self.rules = Rules(True, self.trump)
        self.legal_actions = self.calc_legal_actions()
        return self.current_player, self.encode_state(self.current_player), self.encode_legal_actions()

    def new_initial_state(self, deal=None, current_player=None, index=0):
        self.start_game(deal, current_player)
        return self

    def clone(self):
        return deepcopy(self)

    def child(self, action):
        clone = deepcopy(self)
        clone.step(action)
        return clone

    def deal_cards(self, deal=None):
        if deal is None:
            deck = [Card(rank, suit) for suit in range(c.num_suits)
                         for rank in range(8 - c.num_rank_cards, 8)]
            shuffle(deck)
            i = 0
            n_cards = c.player_cards_2p
            for p in self.players:
                p.deal_hand(deck[i:i + n_cards])
                i += n_cards
            self.players[0].down_cards = deepcopy(deck[i: i+c.down_cards_2p])
            i += c.down_cards_2p
            self.players[1].down_cards = deepcopy(deck[i: i+c.down_cards_2p])
            i += c.down_cards_2p
            self.players[0].up_cards = deepcopy(deck[i: i+c.down_cards_2p])
            i += c.up_cards_2p
            self.players[1].up_cards = deepcopy(deck[i: i+c.down_cards_2p])
        else:
            deal2 = []
            for ind in deal:
                deal2.append(index_to_card(ind))
            i = 0
            for p in self.players:
                p.deal_hand(deal2[i:i + c.player_cards_2p])
                i += c.player_cards_2p
            self.players[0].down_cards = deepcopy(deal2[i:i + c.down_cards_2p])
            i += c.down_cards_2p
            self.players[1].down_cards = deepcopy(deal2[i:i + c.down_cards_2p])
            i += c.down_cards_2p
            self.players[0].up_cards = deepcopy(deal2[i: i + c.down_cards_2p])
            i += c.down_cards_2p
            self.players[1].up_cards = deepcopy(deal2[i: i + c.down_cards_2p])

    def step(self, action):
        card, open = self.decode_action(action)
        if open:
            self.played_up_cards[self.current_player].append(card)
            play = Play(self.current_player, card, self.round_num, len(self.current_trick))
            for ind,car in enumerate(self.players[self.current_player].up_cards):
                if car is not None and car == card:
                    self.players[self.current_player].up_cards[ind] = None
                    self.to_open.append((self.current_player, ind))
        else:
            self.players[self.current_player].hand.remove(card)
            play = Play(self.current_player, card, self.round_num, len(self.current_trick))
        self.current_trick.append(play)
        self.moves.append(play)
        self.current_player = (self.current_player + 1) % self.num_players
        if len(self.current_trick) == self.num_players:
            self.finish_trick()
        payoffs = 0
        done = 1
        if len(self.moves) > c.num_cards - self.num_players:
            done = 0
        self.legal_actions = self.calc_legal_actions()
        return payoffs, done, self.round_num

    def finish_trick(self):
        winner, points, _ = self.rules.judge(self.current_trick)
        self.players[winner].points += points
        self.current_player = winner
        self.current_trick = []
        self.round_num += 1
        for (player, index) in self.to_open:
            if self.players[player].down_cards[index] is not None:
                self.players[player].up_cards[index] = deepcopy(self.players[player].down_cards[index])
                self.players[player].down_cards[index] = None
        self.to_open = []
        return

    def finish_game(self):
        self.over = True
        return

    def information_state_string(self, id=None):
        if id == None:
            return self.encode_state(self.current_player).tostring()
        return self.encode_state(id).tostring()

    def get_state(self, id=None):
        if id is None:
            return self.current_player, self.encode_state(self.current_player), self.encode_legal_actions()
        else:
            return self.current_player, self.encode_state(id), self.encode_legal_actions()

    def get_payoffs(self):
        payoffs = []
        for player in self.players:
            payoffs.append(player.points)
        return payoffs

    def get_current_player_id(self):
        return self.current_player

    def get_current_player_expl(self):
        return self.current_player % 2

    def is_over(self):
        return len(self.moves) == c.num_cards

    def get_legal_actions(self):
        legal = self.legal_actions_mask()
        index, = np.where(np.array(legal) == 1)
        return index

    def legal_actions_mask(self):
        legal = [0] * c.num_cards * 2
        for l in self.calc_legal_actions():
            if index_to_card(l) in list(filter(lambda x: x is not None, self.players[self.current_player].up_cards)):
                legal[c.num_cards + l] = 1
            elif index_to_card(l) in self.players[self.current_player].hand:
                legal[l] = 1
        return legal

    def get_legal_actions_indices(self):
        mask = self.legal_actions_mask()
        index, = np.where(np.array(mask) == 1)
        return index

    def calc_legal_actions(self):
        open = copy(self.players[self.current_player].up_cards)
        open = list(filter(lambda c: c is not None, open))
        hand = self.players[self.current_player].hand + open
        return self.rules.get_legal_actions(self.current_player, hand, self.current_trick)

    def encode_legal_actions(self):
        legal = np.ones(c.num_cards * 2) * -1e32
        for l in self.legal_actions:
            if index_to_card(l) in list(filter(lambda x: x is not None, self.players[self.current_player].up_cards)):
                legal[c.num_cards + l] = 0
            elif index_to_card(l) in self.players[self.current_player].hand:
                legal[l] = 0
        return legal

    def encode_state(self, id):
        if id is None:
            id = self.current_player
        obs = np.zeros(self.state_shape, dtype=float)
        hand = self.players[id].hand
        up_cards1 = self.players[id].up_cards
        up_cards2 = self.players[(id + 1) % self.num_players].up_cards
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
        obs[2] = cards_to_matrix(up_cards1)
        obs[3] = cards_to_matrix(up_cards2)
        obs[4] = cards_to_matrix(self.played_up_cards[id])
        obs[5] = cards_to_matrix(self.played_up_cards[(id + 1) % self.num_players])
        ind = 6
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
        return obs.flatten()

    def decode_action(self, action):
        if action >= c.num_cards:
            card = index_to_card(action - c.num_cards)
            open = True
        else:
            card = index_to_card(action)
            open = False
        return card, open

    def get_state_shape(self):
        return self.state_shape

    def get_perfect_state_shape(self):
        return self.perfect_state_shape

    def get_timestep(self, id=None):
        if id is None:
            id = self.current_player
        obss = [None, None]
        legals = [None, None]
        obss[id] = self.encode_state(None)
        legals[id] = self.get_legal_actions_indices()
        observations = {
            "info_state": obss,
            "legal_actions": legals,
            "current_player": id
        }
        rews = [0,0]
        last = ''
        if self.over:
            rews = self.get_payoffs()
            last = 'last'
        return TimeStep(
            observations=observations,
            rewards=rews,
            discounts=self.discounts,
            step_type=last)

def worker(remote, parent_remote):
    parent_remote.close()
    game = Manil_2p()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            reward, done, info = game.step(data)
            remote.send((reward, done, info))
        elif cmd == 'reset':
            player_id, state, legal_actions = game.start_game()
            remote.send((player_id, state, legal_actions))
        elif cmd == 'getCurrState':
            player_id, state, legal_actions = game.get_state(data)
            remote.send((player_id, state, legal_actions))
        elif cmd == 'getPayoffs':
            payoffs = game.get_payoffs()
            remote.send(payoffs)
        elif cmd == 'close':
            remote.close()
            break
        else:
            print("Invalid command sent by remote")
            break


class VectorizedManil_2p(object):
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

    def get_perfect_state(self, id):
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

class TimeStep(
    collections.namedtuple(
        "TimeStep", ["observations", "rewards", "discounts", "step_type"])):
  """Returned with every call to `step` and `reset`.
  A `TimeStep` contains the data emitted by a game at each step of interaction.
  A `TimeStep` holds an `observation` (list of dicts, one per player),
  associated lists of `rewards`, `discounts` and a `step_type`.
  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.
  Attributes:
    observations: a list of dicts containing observations per player.
    rewards: A list of scalars (one per player), or `None` if `step_type` is
      `StepType.FIRST`, i.e. at the start of a sequence.
    discounts: A list of discount values in the range `[0, 1]` (one per player),
      or `None` if `step_type` is `StepType.FIRST`.
    step_type: A `StepType` enum value.
  """
  __slots__ = ()

  def last(self):
      return self.step_type == "last"

  def current_player(self):
    return self.observations["current_player"]





