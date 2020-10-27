import numpy as np
import collections

import os
import pickle

class BRAgent():
    ''' Implement CFR algorithm
    '''

    def __init__(self, env, policy):
        ''' Initilize Agent
        Args:
            env (Env): Env class
        '''
        self.use_raw = False
        self.env = env
        self._num_of_player = 2

        # A policy is a dict state_str -> action probabilities
        self.opponent_policy = policy
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0

    def traverse_tree(self, probs, player_id):
        ''' Traverse the game tree, get information set
        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value
        Returns:
            state_utilities (list): The expected utilities for all the players
        '''
        if self.env.is_over():
            return self.env.get_payoffs_br()

        current_player = self.env.get_current_player_expl()

        _ , state, legals = self.env.get_state(current_player)
        legal_actions = self.env.get_legal_actions()
        obs = state.tostring()
        _ , action_probs = self.opponent_policy.eval_step(state, legals)

        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            # Keep traversing the child state
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

        # If it is current player, we record the policy and compute regret
        counterfactual_prob = (np.prod(probs[:current_player]) *
                                np.prod(probs[current_player + 1:]))
        _ , state, _ = self.env.get_state(current_player)
        legals = self.env.get_legal_actions()
        self.infosets[obs].append(((state.tostring(), legals), counterfactual_prob))

    def value(self, curr_player, state, legal_actions, this_player):
        """Returns the value of the specified state to the best-responder."""
        if self.env.is_over():
            return self.env.get_payoffs_br()
        elif this_player == curr_player:
            self.infosets = collections.defaultdict(list)
            probs = np.ones(2)
            self.traverse_tree(probs, this_player)
            action = self.best_response_action(this_player, state.tostring())
            q_val = self.get_q_value([0.0, 0.0])
            return q_val[this_player]
        else:
            sum_qval = np.array([0.0, 0.0])
            for a, p in enumerate(self.action_probs(state, legal_actions, self.opponent_policy)):
                q_val = self.get_q_value([0.0, 0.0])
                weighted_qval = np.array([q*p for q in q_val])
                sum_qval += weighted_qval
            return sum_qval[this_player]

    def get_q_value(self, action, q_value):
        if self.env.is_over():
            return self.env.get_payoffs_br()
        _ , state, legals = self.env.get_state(None)
        legal_actions = self.env.get_legal_actions()
        action_probs = self.action_probs(state, legals, self.opponent_policy)
        for act in legal_actions:
            self.env.step(act)
            q_val_out = q_value.copy()
            curr_qval = np.array(self.get_q_value(act, q_value))
            q_val_out += curr_qval * action_probs[act]
            self.env.step_back()
        return q_val_out

    def get_q_value_2(self, q_value):
        if self.env.is_over():
            return self.env.get_payoffs_br()
        current_player, state, legals = self.env.get_state(None)
        if current_player == 0:
            action = self.best_response_action(current_player, state.tostring())
            legal_actions = [action]
            action_probs = np.zeros(8)
            action_probs[action] = 1
        else:
            legal_actions = self.env.get_legal_actions()
            action_probs = self.action_probs(state, legals, self.opponent_policy)
        for act in legal_actions:
            self.env.step(act)
            q_val_out = q_value.copy()
            curr_qval = np.array(self.get_q_value_2(q_value))
            q_val_out += curr_qval * action_probs[act]
            self.env.step_back()
        return q_val_out


    def best_response_action(self, this_player, obs):
        infoset = self.infosets[obs]
        best_act = ""
        max_value = -1000.0
        for each in infoset:
            p, legal_act = each[0]
            cf_p = each[1]
            q_value = [0.0, 0.0]
            for a in legal_act:
                self.env.step(a)
                q_value = self.get_q_value(q_value)
                self.env.step_back()
                tmp_q = cf_p * q_value[this_player]
                if tmp_q > max_value:
                    max_value = tmp_q
                    best_act = a
        return best_act

    def action_probs(self, state, legal_actions, policy):
        ''' Obtain the action probabilities of the current state
        Args:
            state(dictionaty): The state dictionary
            policy (dict): The used policy
        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        _, action_probs = self.opponent_policy.eval_step(state, legal_actions)
        return action_probs

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy
        Args:
            state (numpy.array): State representation
        Returns:
            action (int): Predicted action
        '''
        #probs = self.action_probs(state['obs'].tostring(), state['legal_actions'], self.average_policy)
        this_player = self.env.get_current_player_expl()
        self.infosets = collections.defaultdict(list)
        probs = np.ones(2)
        self.tmp_state = state['obs']
        obs, legal_act = self.get_state(this_player)
        self.traverse_tree(probs, this_player)
        act = self.best_response_action(this_player, state['obs'].tostring())
        return act, []

    def save(self):
        ''' Save model
        '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()

    def remove_illegal(action_probs, legal_actions):
        ''' Remove illegal actions and normalize the
            probability vector
        Args:
            action_probs (numpy.array): A 1 dimention numpy array.
            legal_actions (list): A list of indices of legal actions.
        Returns:
            probd (numpy.array): A normalized vector without legal actions.
        '''
        probs = np.zeros(action_probs.shape[0])
        probs[legal_actions] = action_probs[legal_actions]
        if np.sum(probs) == 0:
            probs[legal_actions] = 1 / len(legal_actions)
        else:
            probs /= sum(probs)
        return probs