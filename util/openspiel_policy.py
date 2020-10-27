# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Representation of a policy for a game.
This is a standard representation for passing policies into algorithms,
with currently the following implementations:
  TabularPolicy - an explicit policy per state, stored in an array
    of shape `(num_states, num_actions)`, convenient for tabular policy
    solution methods.
  UniformRandomPolicy - a uniform distribution over all legal actions for
    the specified player. This is computed as needed, so can be used for
    games where a tabular policy would be unfeasibly large.
The main way of using a policy is to call `action_probabilities(state,
player_id`), to obtain a dict of {action: probability}. `TabularPolicy`
objects expose a lower-level interface, which may be more efficient for
some use cases.
"""

from copy import deepcopy
import itertools
import numpy as np
import pickle
import config as config
from manil.util import generate_game_states

from manil.game2 import Manil2
from manil.card import Card

class Policy(object):
  """Base class for policies.
  A policy is something that returns a distribution over possible actions
  given a state of the world.
  Attributes:
    game: the game for which this policy applies
    player_ids: list of player ids for which this policy applies; each in the
      interval [0..game.num_players()-1].
  """

  def __init__(self, game, player_ids):
    """Initializes a policy.
    Args:
      game: the game for which this policy applies
      player_ids: list of player ids for which this policy applies; each should
        be in the range 0..game.num_players()-1.
    """
    self.game = game
    self.player_ids = player_ids

  def action_probabilities(self, state, player_id=None):
    """Returns a dictionary {action: prob} for all legal actions.
    IMPORTANT: We assume the following properties hold:
    - All probabilities are >=0 and sum to 1
    - TLDR: Policy implementations should list the (action, prob) for all legal
      actions, but algorithms should not rely on this (yet).
      Details: Before May 2020, only legal actions were present in the mapping,
      but it did not have to be exhaustive: missing actions were considered to
      be associated to a zero probability.
      For example, a deterministic state-poliy was previously {action: 1.0}.
      Given this change of convention is new and hard to enforce, algorithms
      should not rely on the fact that all legal actions should be present.
    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.
    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state.
    """
    raise NotImplementedError()

  def __call__(self, state, player_id=None):
    """Turns the policy into a callable.
    Args:
      state: The current state of the game.
      player_id: Optional, the player id for whom we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.
    Returns:
      Dictionary of action: probability.
    """
    return self.action_probabilities(state, player_id)

  def to_tabular(self):
    """Returns a new `TabularPolicy` equivalent to this policy."""
    tabular_policy = TabularPolicy(self.game, self.player_ids)
    for index, state in enumerate(tabular_policy.states):
      tabular_policy.action_probability_array[index, :] = 0
      for action, probability in self.action_probabilities(state).items():
        tabular_policy.action_probability_array[index, action] = probability
    return tabular_policy


class TabularPolicy(Policy):
  """Policy implementation where the policy is in explicit tabular form.
  In addition to implementing the `Policy` interface, this class exposes
  details of the policy representation for easy manipulation.
  The states are guaranteed to be grouped by player, which can simplify
  code for users of this class, i.e. `action_probability_array` contains
  states for player 0 first, followed by states for player 1, etc.
  The policy uses `state.information_state` as the keys if available, otherwise
  `state.observation`.
  Usages:
  - Set `policy(info_state, action)`:
  ```
  tabular_policy = TabularPolicy(game)
  info_state_str = state.information_state_string(<optional player>)
  state_policy = tabular_policy.policy_for_key(info_state_str)
  state_policy[action] = <value>
  ```
  - Set `policy(info_state)`:
  ```
  tabular_policy = TabularPolicy(game)
  info_state_str = state.information_state_string(<optional player>)
  state_policy = tabular_policy.policy_for_key(info_state_str)
  state_policy[:] = <list or numpy.array>
  ```
  Attributes:
    action_probability_array: array of shape `(num_states, num_actions)`, where
      `action_probability_array[s, a]` is the probability of choosing action `a`
      when at state `s`.
    state_lookup: `dict` mapping state key string to index into the
      `tabular_policy` array. If information state strings overlap, e.g. for
      different players or if the information state string has imperfect recall,
      then those states will be mapped to the same policy.
    legal_actions_mask: array of shape `(num_states, num_actions)`, each row
      representing which of the possible actions in the game are valid in this
      particular state, containing 1 for valid actions, 0 for invalid actions.
    states_per_player: A `list` per player of the state key strings at which
      they have a decision to make.
    states: A `list` of the states as ordered in the `action_probability_array`.
    state_in: array of shape `(num_states, state_vector_size)` containing the
      normalised vector representation of each information state. Populated only
      for games which support information_state_tensor(), and is None otherwise.
    game_type: The game attributes as returned by `Game::GetType`; used to
      determine whether to use information state or observation as the key in
      the tabular policy.
  """

  def __init__(self, game, states, players=None):
    """Initializes a uniform random policy for all players in the game."""
    players = range(4)
    super(TabularPolicy, self).__init__(game, range(game.num_players))

    # Get all states in the game at which players have to make decisions.
    all_states = get_all_states(
        game,
        states,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False)

    # Assemble legal actions for every valid (state, player) pair, keyed by
    # information state string.
    self.state_lookup = {}
    self.states_per_player = [[] for _ in range(game.num_players)]
    self.states = []
    legal_actions_list = []
    state_in_list = []
    for player in players:
      # States are ordered by their history.
      for _, state in sorted(all_states.items(), key=lambda pair: pair[0]):
        if player == state.get_current_player_id():
          legal_actions = state.legal_actions_mask()
          if any(legal_actions):
            key = self._state_key(state, player)
            if key not in self.state_lookup:
              state_index = len(legal_actions_list)
              self.state_lookup[key] = state_index
              legal_actions_list.append(legal_actions)
              self.states_per_player[player].append(key)
              self.states.append(state)
          else:
              self.states.append(state)

    # Put legal action masks in a numpy array and create the uniform random
    # policy.
    self.state_in = None
    if state_in_list:
      self.state_in = np.array(state_in_list)
    self.legal_actions_mask = np.array(legal_actions_list)
    self.action_probability_array = (
        self.legal_actions_mask /
        np.sum(self.legal_actions_mask, axis=-1, keepdims=True))

  def _state_key(self, state, player):
    """Returns the key to use to look up this (state, player) pair."""
    return state.information_state_string(player)

  def action_probabilities(self, state, player_id=None):
    """Returns an {action: probability} dict, covering all legal actions."""
    probability = self.policy_for_key(self._state_key(state, player_id))
    legal_actions = (state.get_legal_actions())
    return {action: probability[action] for action in legal_actions}

  def action_probs(self, obs, encoded_legal_actions, legal_actions):
      probability = self.policy_for_key(obs.tostring())
      return [(a, probability[a]) for a in legal_actions]

  def state_index(self, state):
    """Returns the index in the TabularPolicy associated to `state`."""
    return self.state_lookup[self._state_key(state, state.get_current_player_id())]

  def get_all_states(self):
      return self.states

  def policy_for_key(self, key):
    """Returns the policy as a vector given a state key string.
    Args:
      key: A key for the specified state.
    Returns:
      A vector of probabilities, one per action. This is a slice of the
      backing policy array, and so slice or index assignment will update the
      policy. For example:
      ```
      tabular_policy.policy_for_key(s)[:] = [0.1, 0.5, 0.4]
      ```
    """
    return self.action_probability_array[self.state_lookup[key]]

  def __copy__(self, copy_action_probability_array=True):
    """Returns a shallow copy of self.
    Most class attributes will be pointers to the copied object's attributes,
    and therefore altering them could lead to unexpected behavioural changes.
    Only action_probability_array is expected to be modified.
    Args:
      copy_action_probability_array: Whether to also include
        action_probability_array in the copy operation.
    Returns:
      Copy.
    """
    result = TabularPolicy.__new__(TabularPolicy)
    result.state_lookup = self.state_lookup
    result.legal_actions_mask = self.legal_actions_mask
    result.state_in = self.state_in
    result.state_lookup = self.state_lookup
    result.states_per_player = self.states_per_player
    result.states = self.states
    result.game = self.game
    result.player_ids = self.player_ids
    if copy_action_probability_array:
      result.action_probability_array = np.copy(self.action_probability_array)
    return result

def _get_subgames_states(state, all_states, depth_limit, depth,
                       include_terminals, include_chance_states,
                       stop_if_encountered):
  """Extract non-chance states for a subgame into the all_states dict."""
  if state.is_over():
      if include_terminals:
          # Include if not already present and then terminate recursion.
          state_str = state.information_state_string(None)
          if state_str not in all_states:
              all_states[state_str] = state.clone()
      return

  if depth > depth_limit >= 0:
      return

  # Add only if not already present
  state_str = state.information_state_string(None)
  if state_str not in all_states:
      all_states[state_str] = state.clone()
  else:
      # We already saw this one. Stop the recursion if the flag is set
      if stop_if_encountered:
          return

  for action in state.get_legal_actions():
      state_for_search = state.child(action)
      _get_subgames_states(state_for_search, all_states, depth_limit, depth + 1,
                           include_terminals, include_chance_states, stop_if_encountered)

def get_all_states(game,
                   root_states,
                 depth_limit=-1,
                 include_terminals=True,
                 include_chance_states=False,
                 stop_if_encountered=False):
  """Gets all states in the game, indexed by their string representation.
  For small games only! Useful for methods that solve the  games explicitly,
  i.e. value iteration. Use this default implementation with caution as it does
  a recursive tree walk of the game and could easily fill up memory for larger
  games or games with long horizons.
  Currently only works for sequential games.
  Arguments:
    game: The game to analyze, as returned by `load_game`.
    depth_limit: How deeply to analyze the game tree. Negative means no limit, 0
      means root-only, etc.
    include_terminals: If True, include terminal states.
    include_chance_states: If True, include chance node states.
    to_string: The serialization function. We expect this to be
      `lambda s: s.history_str()` as this enforces perfect recall, but for
        historical reasons, using `str` is also supported, but the goal is to
        remove this argument.
    stop_if_encountered: if this is set, do not keep recursively adding states
      if this state is already in the list. This allows support for games that
      have cycles.
  Returns:
    A `dict` with `to_string(state)` keys and `pyspiel.State` values containing
    all states encountered traversing the game tree up to the specified depth.
  """
  # Get the root state.
  '''
  states = generate_game_states(game, different_starters=True)
  print("Possible dealings: ", len(states))
  '''
  all_states = dict()

  # Then, do a recursive tree walk to fill up the map.
  for state in root_states:
      _get_subgames_states(
          state=state,
          all_states=all_states,
          depth_limit=depth_limit,
          depth=0,
          include_terminals=include_terminals,
          include_chance_states=include_chance_states,
          stop_if_encountered=stop_if_encountered)

  print("Total number of infosets: ", len(all_states))

  if not all_states:
      raise ValueError("GetSubgameStates returned 0 states!")

  return all_states

def policy_value(state, policies):
  """Returns the expected values for the state for players following `policies`.
  Computes the expected value of the`state` for each player, assuming player `i`
  follows the policy given in `policies[i]`.
  Args:
    state: A `pyspiel.State`.
    policies: A `list` of `policy.Policy` objects, one per player.
  Returns:
    A `numpy.array` containing the expected value for each player.
  """
  num_players = len(policies)
  if state.is_over():
    values = np.array(state.get_payoffs())
  else:
      player = state.get_current_player_id()
      values = np.zeros(shape=num_players)
      for action, probability in policies[player].action_probabilities(state).items():
        if probability > 0:
            child = state.child(action)
            values += probability * policy_value(child, policies)
  return values