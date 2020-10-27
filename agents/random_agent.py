import numpy as np

class RandomAgent(object):

    def __init__(self, action_num):
        self.action_num = action_num

    def step(self, legal_actions):
        res = []
        probs = []
        for legal in legal_actions:
            index, = np.where(legal == 0)
            res.append(np.random.choice(index))
            action_probs = [1.0 / len(index) if a == 0 else 0.0 for a in legal]
            probs.append(action_probs)
        return res, probs

    def step_1(self, legal_actions):
        index, = np.where(legal_actions == 0)
        action_probs = [1.0 / len(index) if a == 0 else 0.0 for a in legal_actions]
        return np.random.choice(index), action_probs

    def eval_step(self, state, legal_actions):
        index, = np.where(legal_actions == 0)
        action_probs = [1.0 / len(index) if a == 0 else 0.0 for a in legal_actions]
        return np.random.choice(index), action_probs

    def eval_step2(self, state, legal_actions):
        index, = np.where(legal_actions > 0)
        action_probs = [1.0 / len(index) if a == 0 else 0.0 for a in legal_actions]
        return np.random.choice(index), action_probs

    def action_probs(self, state, encoded_legal_actions, legal_actions):
        return [(a, 1.0 / len(legal_actions)) for a in legal_actions]

    def action_probabilities(self, state):
        legal_actions = state.get_legal_actions()
        return {action: 1.0 / len(legal_actions) for action in legal_actions}

