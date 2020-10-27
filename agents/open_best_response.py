import collections
import pickle

class BestResponsePolicy():
  """Computes the best response to a specified strategy."""

  def __init__(self, game, player_id, policy, root_state=None):
    """Initializes the best-response calculation.
    Args:
      game: The game to analyze.
      player_id: The player id of the best-responder.
      policy: A `policy.Policy` object.
      root_state: The state of the game at which to start analysis. If `None`,
        the game root state is used.
    """
    self._num_players = 2
    self._player_id = player_id
    self._policy = policy
    self.infosets = collections.defaultdict(list)
    if root_state is None:
      root_state = game.new_initial_state()
    self.info_sets(root_state)

  def all_info_sets(self, states):
    self.infosets = collections.defaultdict(list)
    for i, state in enumerate(states):
      for s, p in self.decision_nodes(state):
        self.infosets[s.information_state_string(None)].append((s, p))
    self.infosets = dict(self.infosets)


  def info_sets(self, state):
    """Returns a dict of infostatekey to list of (state, cf_probability)."""
    for s, p in self.decision_nodes(state):
      self.infosets[s.information_state_string(None)].append((s, p))
    self.infosets = dict(self.infosets)

  def decision_nodes(self, parent_state):
    """Yields a (state, cf_prob) pair for each descendant decision node."""
    if not parent_state.is_over():
      if parent_state.get_current_player() == self._player_id:
        yield (parent_state, 1.0)
      for action, p_action in self.transitions(parent_state):
        for state, p_state in self.decision_nodes(parent_state.child(action)):
          yield (state, p_state * p_action)

  def transitions(self, state):
    """Returns a list of (action, cf_prob) pairs from the specifed state."""
    legal_actions = state.get_legal_actions()
    if state.get_current_player() == self._player_id:
      # Counterfactual reach probabilities exclude the best-responder's actions,
      # hence return probability 1.0 for every action.
      return [(action, 1.0) for action in legal_actions]
    else:
      return self.action_probs(state, legal_actions)

  def value(self, state):
    """Returns the value of the specified state to the best-responder."""
    if state.is_over():
      return state.player_return(self._player_id)
    elif state.get_current_player() == self._player_id:
      action = self.best_response_action(
          state.information_state_string(None))
      return self.q_value(state, action)
    else:
      return sum(p * self.q_value(state, a) for a, p in self.transitions(state))

  def q_value(self, state, action):
    """Returns the value of the (state, action) to the best-responder."""
    return self.value(state.child(action))

  def best_response_action(self, infostate):
    """Returns the best response for this information state."""
    infoset = self.infosets[infostate]
    # Get actions from the first (state, cf_prob) pair in the infoset list.
    # Return the best action by counterfactual-reach-weighted state-value.
    return max(
        infoset[0][0].get_legal_actions(),
        key=lambda a: sum(cf_p * self.q_value(s, a) for s, cf_p in infoset))

  def action_probs(self, state, legal_actions):
    obs = state.encode_state(state.get_current_player_id())
    encoded_legal_actions = state.encode_legal_actions(legal_actions)
    return self._policy.action_probs(obs, encoded_legal_actions, legal_actions)