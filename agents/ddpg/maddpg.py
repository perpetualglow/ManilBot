import torch.optim as optim

from agents.ddpg.ddpg import Actor, Critic
from agents.ddpg.replaybuffer import ReplayBuffer
from agents.ppo.config import *
from agents.util import *


class MA_DDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, state_shape, perfect_state_shape, num_actions, num_agents):
        self.device = DEVICE
        self.state_shape = state_shape
        self.perfect_state_shape = perfect_state_shape
        self.num_actions = num_actions
        self.num_players = num_agents

        # self.agents = [Single_DDPG_Agent(state_shape, num_actions, num_agents, i) for i in range(num_agents)]

        layer_dims = [np.prod(self.perfect_state_shape)] + [512,256,256,32]

        self.local_critics = [Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
                              Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
                              Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
                              Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE)
                              ]
        self.target_critics = [Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
                               Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
                               Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
                               Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE)]
        self.critic_optimizers = [optim.Adam(self.local_critics[0].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
                                  optim.Adam(self.local_critics[1].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
                                  optim.Adam(self.local_critics[2].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
                                  optim.Adam(self.local_critics[3].parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                                  ]
        self.local_critic = Critic(perfect_state_shape, num_actions, layer_dims, DEVICE).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        hard_update(self.target_critics[0], self.local_critics[0])
        hard_update(self.target_critics[1], self.local_critics[1])

        layer_dims = [np.prod(self.state_shape)] + [256,256,256]
        self.local_actors = [
            Actor(state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
            Actor(state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
            Actor(state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
            Actor(state_shape, num_actions, layer_dims, DEVICE).to(DEVICE)
        ]
        self.target_actors = [
            Actor(state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
            Actor(state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
            Actor(state_shape, num_actions, layer_dims, DEVICE).to(DEVICE),
            Actor(state_shape, num_actions, layer_dims, DEVICE).to(DEVICE)
        ]
        self.actor_optimizers = [
            optim.Adam(self.local_actors[0].parameters(), lr=0.001),
            optim.Adam(self.local_actors[1].parameters(), lr=0.001),
            optim.Adam(self.local_actors[2].parameters(), lr=0.001),
            optim.Adam(self.local_actors[3].parameters(), lr=0.001),
        ]

        hard_update(self.target_actors[0], self.local_actors[0])
        hard_update(self.target_actors[1], self.local_actors[1])
        hard_update(self.target_actors[2], self.local_actors[2])
        hard_update(self.target_actors[3], self.local_actors[3])
        self.MSELoss = torch.nn.MSELoss()

        self.memory = ReplayBuffer()

    def add(self, obs, perfect_state, legals, action, reward, next_perfect_state, done):
        self.memory.add(obs, perfect_state, legals, action, reward, next_perfect_state, done)

    def step(self, obs, legal_actions, agent, explore=False):
        agent = 0
        actions = self.local_actors[agent](obs, legal_actions)
        if explore:
            action = gumbel_softmax(actions, hard=True)
        else:
            action = onehot_from_logits(actions)
        return action

    def step_target(self, obs, legal_actions, agent, explore=False):
        agent = 0
        actions = self.target_actors[agent](obs, legal_actions)
        if explore:
            action = gumbel_softmax(actions, hard=True)
        else:
            action = onehot_from_logits(actions)
        return action

    def eval_step(self, obs, legal_actions, agent, explore=False):
        agent = 0
        with torch.no_grad():
            self.local_actors[agent].eval()
            actions = self.local_actors[agent](obs, legal_actions)
            if explore:
                action = gumbel_softmax(actions, hard=True)
            else:
                action = onehot_from_logits(actions)
            self.local_actors[agent].train()
            return action

    def sim_step(self, obs, legal_actions, agent):
        agent = 0
        obs = torch.from_numpy(np.array([obs])).float().to(self.device)
        legal_actions = torch.from_numpy(np.array([legal_actions])).float().to(self.device)
        actions = self.local_actors[agent](obs, legal_actions)
        return torch.argmax(actions)

    def update(self, agent, actor):
        agent2 = agent % 2
        critic = self.local_critics[agent2]
        target_critic = self.target_critics[agent2]
        critic_optimizer = self.critic_optimizers[agent2]

        obs, perfect_state_obs, perfect_state, legals, action, reward, next_perfect_state, done = self.memory.sample(agent, 1024)
        next_obs = []
        next_actions = []
        for s in next_perfect_state:
            s2 = s.clone()
            perf_state = torch.from_numpy(s2.encode_state_perfect(0).copy())
            next_obs.append(perf_state)
            n_act = [0] * self.num_players
            if not s2.is_over():
                for i in range(self.num_players):
                    player_id, state, legal_actions = s2.get_state(None)
                    a = self.step_target(torch.from_numpy(state.copy()).float(), torch.from_numpy(legal_actions.copy()).float(), i)
                    n_act[player_id] = a
                    action_int = torch.argmax(a, dim=-1).float().cpu().numpy()
                    s2.step(action_int)
                next_actions.append(torch.cat(n_act, dim=-1).float())
            else:
                next_actions.append(torch.zeros(self.num_players*self.num_actions))
        next_obs = torch.stack(next_obs).float()
        next_actions = torch.stack(next_actions).float()
        perfect_state_obs2 = torch.stack(perfect_state_obs).float()
        action2 = torch.stack(action).float()
        trgt_vf_in = torch.cat((next_obs, next_actions), dim=-1).float()
        vf_in = torch.cat((perfect_state_obs2, action2), dim=-1).float()

        # actual_value = critic(vf_in)
        # target = target_critic(trgt_vf_in)
        actual_value = critic(perfect_state_obs2, action2)
        target = target_critic(next_obs, next_actions)
        target_value = (torch.FloatTensor(reward).view(-1,1) + 1 *
                        target *
                        (torch.FloatTensor(done).view(-1, 1)))
        vf_loss = self.MSELoss(actual_value, target_value.detach())
        print("Value Loss ", vf_loss)
        critic_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
        critic_optimizer.step()

        if actor:
            # obs = torch.stack(obs).float()
            # legals = torch.stack(legals).float()
            # curr_pol_out = actor(obs, legals)
            # curr_pol_out = torch.exp(curr_pol_out)
            # act_obs = []
            act_actions = [[],[], [], []]
            perfect_state_obss = [[],[],[], []]
            for s in perfect_state:
                s2 = s.clone()
                prev_perfect_state_obs = torch.from_numpy(s2.encode_state_perfect(0).copy())
                n_act2 = [0] * self.num_players
                if not s2.is_over():
                    player_id, state, legal_actions = s2.get_state(None)
                    # ind = (agent - player_id) % self.num_players
                    ind = 0
                    for i in range(self.num_players):
                        player_id, state, legal_actions = s2.get_state(None)
                        state = torch.from_numpy(state.copy()).float()
                        legal_actions = torch.from_numpy(legal_actions.copy()).float()
                        if player_id == agent:
                            a = self.step(state, legal_actions, i, explore=True)
                            n_act2[player_id] = a.float()
                        # elif (player_id + 2) % self.num_players == agent:
                        #     a = self.step(state, legal_actions, i, explore=True)
                        #     n_act2[i] = a.float()
                        else:
                            a = self.step(state, legal_actions, i).detach()
                            n_act2[player_id] = a.float()
                        action_int = torch.argmax(a, dim=-1).float().cpu().numpy()
                        s2.step(action_int)
                    act_actions[ind].append(torch.cat(n_act2, dim=-1).float())
                    perfect_state_obss[ind].append(prev_perfect_state_obs)
            # # act_obs = torch.stack(act_obs)
            # act_actions = torch.stack(act_actions)
            # vf_in2 = torch.cat((perfect_state_obs, act_actions), dim=-1).float()
            # # vf_in2 = torch.cat((perfect_state_obs, action), dim=-1).float()
            # pol_loss = -critic(vf_in2).mean()
            # # pol_loss += (curr_pol_out**2).mean()
            # actor_optimizer.zero_grad()
            # pol_loss.backward()
            # actor_optimizer.step()
            for i in range(1):
                act_act = torch.stack(act_actions[i]).float()
                perf_state_o = torch.stack(perfect_state_obss[i]).float()
                # vf_in2 = torch.cat((perf_state_o, act_act), dim=-1).float()
                pol_loss = -critic(perf_state_o, act_act).mean()
                self.actor_optimizers[i].zero_grad()
                pol_loss.backward()
                self.actor_optimizers[i].step()



    def update_all_targets(self):
        soft_update(self.target_critics[0], self.local_critics[0], 0.01)
        soft_update(self.target_critics[1], self.local_critics[1], 0.01)
        soft_update(self.target_actors[0], self.local_actors[0], 0.01)
        soft_update(self.target_actors[1], self.local_actors[1], 0.01)
        soft_update(self.target_actors[2], self.local_actors[2], 0.01)
        soft_update(self.target_actors[3], self.local_actors[3], 0.01)




