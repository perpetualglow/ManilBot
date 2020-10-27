import time
import pickle
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch
from manil.game import Manil, VectorizedManil
from manil.game2 import Manil2
from manil.two_player_manil import Manil_2p, VectorizedManil_2p
from environment import Environment
from environment2 import Environment2
from agents.random_agent import RandomAgent
from agents.ppo.ppo import PPO
from agents.ppo.ppo2 import PPO2
from agents.ppo.ppo_agent import PPOAgent
from agents.CFR_agent import CFRSolver
import config as c
from util.openspiel_policy import policy_value
from manil.util import generate_game_states, generate_game_states_2p
from agents.ddpg.maddpg import MA_DDPG
from agents.nfsp.nfsp import NFSPAgent

if __name__ == '__main__':
    # vectorized_game = VectorizedManil(c.NUM_CONCURRENT_GAMES)
    game = Manil()
    vectorized = VectorizedManil(c.NUM_CONCURRENT_GAMES)
    vectorized_2p = VectorizedManil_2p(c.NUM_CONCURRENT_GAMES)
    game_2p = Manil_2p()
    env = Environment(game_2p, vectorized_2p)
    writer = SummaryWriter()
    random_agent = RandomAgent(c.num_cards)
    env2 = Environment2(game_2p, None)

    network1 = PPO(game_2p.get_state_shape(), game_2p.get_perfect_state_shape(), 2 * c.num_cards, c.network_shape)
    # network2 = PPO2(game.get_state_shape(),c.num_cards, [128,128])
    # network3 = PPO2(game.get_state_shape(), c.num_cards, [128, 128])
    network_2p = PPO(game_2p.get_state_shape(), game_2p.get_state_shape(), c.num_cards * 2, c.network_shape)
    network2_2p = PPO(game_2p.get_state_shape(), game_2p.get_state_shape(), c.num_cards * 2, c.network_shape)
    ddpg = MA_DDPG(c.state_shape, c.perfect_state_shape, c.num_cards, c.num_players)
    agent1 = PPOAgent(network1, writer=None)
    agent3 = PPOAgent(network2_2p, writer=None)
    agent2 = RandomAgent(c.num_cards)
    game = Manil_2p()
    nfsps = []
    for i in range(c.num_players):
        agent = NFSPAgent(scope='nfsp' + str(i),
                          action_num=c.num_cards * 2,
                          state_shape=c.state_shape_2p,
                          hidden_layers_sizes=[128,128],
                          min_buffer_size_to_learn=1000,
                          q_replay_memory_init_size=1000,
                          train_every=2,
                          q_train_every=2,
                          q_mlp_layers=[128,128],
                          device=torch.device('cpu'))
        nfsps.append(agent)
    env2.set_agents(nfsps)
    env2.set_eval_agents([nfsps[0], random_agent])
    agents = [agent1, agent2 ,agent1, agent2]
    eval_agents = [agent1, agent2, agent1, agent2]
    old_networks = []

    env.set_eval_agents(eval_agents)
    env.set_agents(agents)



    #env.full_exploitability()

    #env.br2()

    def run_policy(total_episodes):
        env2.run_policy()

    def run_nfsp(total_episodes):
        for ep in range(total_episodes + 1):
            env2.run_nfsp()
            if ep % 5000 == 0 and ep > 0:
                scores = env2.simulate_against_random(10000)
                print("Episode ", ep, " score: ", scores)


    def build_complete_br(states_path, br_path):
        game = Manil2()
        generate_game_states(game,states_path=states_path)
        env.build_complete_br(states_path, br_path)

    def train(total_episodes, save_path):
        # with open("data/cfr_agents/cfr_2p_8c_2h_2s_trump_50.pkl", 'rb') as file_handler:
        #     cfr_solver = pickle.load(file_handler)
        # with open("data/states/s_2p_8c_2h_2s_trump_50.pkl", 'rb') as file_handler:
        #     states = pickle.load(file_handler)
        start = time.time()
        scores = 0
        ag = None
        for i in range(total_episodes+1):
            env.train_against_self()
            # if i % 100 == 0:
            #     print("Update agents")
            #     ag = env.update_agents()
            # if i % 100 == 0 and i > 0:
            #     pth = save_path + str(i)
            #     agent1.save(pth)
            #     print("Model Saved, iteration: ", i)
            if i % 50 == 0 and i > 0:

                score = env.simulate_against_random(5000)
                score = score
                # writer.add_scalar("Evaluation/Random_agent", score, i)
                end = time.time()
                print("Iteration: ", i, " Score: ", score, " Time: ", end-start)
                start = time.time()
            # if i % 1000 == 0 and i > 0:
            #     res = [0, 0]
            #     for s in states:
            #         val1 = policy_value(s, [cfr_solver, agent1])
            #         val2 = policy_value(s, [agent1, cfr_solver])
            #         x = game_2p.utility * ((val1[1]) / (val1[0] + val1[1]))
            #         y = game_2p.utility * ((val2[0]) / (val2[0] + val2[1]))
            #         res[0] += x + y
            #     res[0] = res[0] / (2*len(states))
            #     print("Versus CFR: ", res[0])
            #     writer.add_scalar("Evaluation/CFR", (res[0] - game_2p.utility / 2), i)

        agent1.save(save_path)
        writer.close()

    def eval_saved_ppo_cfr(cfr_path, ppo_paths, states):
        with open(cfr_path, 'rb') as file_handler:
            cfr_solver = pickle.load(file_handler)

        average_policy = cfr_solver.average_policy()
        for i, path in enumerate(ppo_paths):
            ppo = agent1.load(path + str((i+1)*10))
            policy_values = [0, 0]
            states = generate_game_states(game)
            for state in states:
                if game.num_players == 2:
                    val1 = policy_value(state, [average_policy, ppo])
                    val2 = policy_value(state, [ppo, average_policy])
                    policy_values[0] += val1[1]
                    policy_values[0] += val2[0]
            print("Iteration ", i, " ",(policy_values[0]) / (len(states) * 2))


    def approx_exploitability(total_episodes, agent):
        env.br2(total_episodes, agent)

    def full_exploitability(agent):
        game = Manil2()
        states = generate_game_states(game)
        env.full_exploitability(states, agent)

    def build_cfr_solver(game, num_iterations, path, all_states=None):
        if all_states is None:
            if game.num_players == 2:
                states = generate_game_states(game, different_starters=False)
            else:
                states = generate_game_states(game, different_starters=True)
        else:
            states = all_states
        cfr_solver = CFRSolver(game, states)
        for ind, state in enumerate(states):
            print("Iteration ", ind, " of ", len(states))
            for i in range(num_iterations):
                cfr_solver.evaluate_and_update_policy(state)
        with open(path, 'wb') as file_handler:
            pickle.dump(cfr_solver, file_handler)
            print("CFR saved!")
        return cfr_solver

    def eval_cfr(game, cfr_path, opponent_policy=None):
        with open(cfr_path, 'rb') as file_handler:
            cfr_solver = pickle.load(file_handler)

        if opponent_policy is None:
            opponent_policy = cfr_solver.average_policy()
        policy_values = [0,0]
        states = generate_game_states(game)
        for state in states:
            if game.num_players == 2:
                val1 = policy_value(state, [cfr_solver, opponent_policy])
                val2 = policy_value(state, [opponent_policy, cfr_solver])
                policy_values[0] += val1[1]
                policy_values[0] += val2[0]
                #x1 = policy_value(state, [opponent_policy, average_policy])
                #x2 = policy_value(state, [average_policy, opponent_policy])
                #average_policy_values = [(x1[0] + x2[1]) / 2, x1[1] + x2[0]]
            else:
                average_policy_values = policy_value(state, [cfr_solver, opponent_policy, cfr_solver, opponent_policy])
        print((policy_values[0])/(len(states) *2))

    def eval_policies(game, policies, states=None):
        if states is None:
            states = generate_game_states(game)
        policy_values = np.array([0]*states[0].num_players, dtype=float)
        for i, state in enumerate(states):
            if i % 50 == 0:
                print("Iteration ", i)
            average_policy_values = policy_value(state, policies)
            policy_values += average_policy_values
        return policy_values / len(states)

    def print_cfr(cfr_path, output_path):
        with open(cfr_path, 'rb') as file_handler:
            cfr_solver = pickle.load(file_handler)
        policy = cfr_solver.average_policy()
        all_states = policy.get_all_states()
        f = open(output_path, "w")
        f.write("Utility sum: " + str(all_states[0].utility))
        f.write("Total amount of infosets (not including terminal states): " + str(len(all_states)) + "\n")
        max_index = 0
        for state in all_states:
            if state.index > max_index:
                max_index = state.index
        f.write("Number of possible deals: " + str(max_index + 1) + "\n")
        for i in range(max_index + 1):
            states = list(filter(lambda s: (s.index == i), all_states))
            states = sorted(states, key=lambda s: len(s.moves))
            f.write("\n" + "Deal " + str(i) + "\n")
            for s in states:
                if not s.is_over():
                    player = s.get_current_player_id()
                    table_card = ""
                    if len(s.current_trick) > 0:
                        table_card = str(s.current_trick[0])
                    prob = policy.policy_for_key(s.information_state_string())
                    probs = []
                    for c in s.players[player].hand:
                        probs.append(prob[c.index])
                    string = ""
                    string += "Player: " + str(player) + " Hand: " + str(s.players[player].hand) + "  Table card: " \
                              + table_card + "    Strategy:   " + str(probs) \
                              + "  Points: " + str(s.players[player].points) + '\n'
                    f.write(string)

        f.close()

    def load_cfr(cfr_path):
        with open(cfr_path, 'rb') as file_handler:
            cfr_solver = pickle.load(file_handler)
        return cfr_solver

    # run_nfsp(500000)
    run_policy(500000)

    # env.run_ddpg()

    #
    # game = Manil2()
    # with open("data/states/state_4p_2s_8c.pkl", 'rb') as file_handler:
    #     states = pickle.load(file_handler)
    # print(states[0])
    # with open("data/cfr_agents/cfr_4p_2s_8c_m", 'rb') as file_handler:
    #     cfr = pickle.load(file_handler)
    # x = eval_policies(game, [cfr, agent1, cfr, agent1], states[0:500])
    # print(x)
    #
    #agent1.load("data/ppo_agents/ppo_agent_4p_4s_12c_m")
    #approx_exploitability(10, agent1)



    # game_2p = Manil_2p()
    # states = generate_game_states_2p(game_2p, states_path="data/states/s_2p_8c_2h_2s_trump_50.pkl", num_states=50)
    # build_cfr_solver(game_2p, 50, "data/cfr_agents/cfr_2p_8c_2h_2s_trump_50.pkl", all_states=states)
    train(5000, "data/ppo_agents/ppo_agent_2p_8c_2h_trump_")
    # env.run_dqn()
    # run_nfsp(100000)


    # shuffle(states)
    # with open("data/states/state_4p_2s_12c.pkl", 'wb') as file_handler:
    #     pickle.dump(states, file_handler)
    # build_cfr_solver(game, 50, "data/cfr_agents/cfr_4p_2s_8c_m",states[0:5])
    # print_cfr("data/cfr_agents/cfr_4p_2s_8c_m", "data/cfr_4p_8c_2s_m.txt")
    # cfr = load_cfr("data/cfr_agents/cfr_4p_2s_8c_m")
    # x = eval_policies(game, [cfr, random_agent, cfr, random_agent], states[0:500])
    # print(x)
    #agent1.load("data/ppo_agents/ppo_agent_2p_2s_6c_m")
    #eval_cfr(game, "data/cfr_agents/cfr_2p_2s_6c_m", agent1)
    #res = eval_policies(game, [agent1, random_agent])
    #cfr = load_cfr("data/cfr_agents/cfr_2p_2s_6c_m")
    #full_exploitability(cfr)


    #full_exploitability(random_agent)


    # '''
    # game = Manil2()
    # states = generate_game_states(game)
    # shuffle(states)
    # build_cfr_solver(game, 10, "data/cfr_agents/cfr_2p_1s_4c_m", states)
    # eval_cfr(game, "data/cfr_agents/cfr_2p_1s_4c_m", random_agent)
    # '''
    #with open("data/cfr_agents/cfr_2p_2s_6c_m", 'rb') as file_handler:
     #   cfr_solver = pickle.load(file_handler)
    #approx_exploitability(100, random_agent)


    #game = Manil2()
    #cfr = build_cfr_solver(game, 100, "data/cfr_agents/cfr_2p_2s_6c_m")
    #agent1.load("data/ppo_agents/ppo_agent_2p_2s_6c_m")
    #eval_cfr(game, "data/cfr_agents/cfr_2p_2s_6c_m", agent1)
    #print_cfr("data/cfr_agents/cfr_2p_2s_6c_m", "data/cfr_2p_6c_2s_m.txt")
    #build_complete_br('data/states/state_4_1_8.pkl', 'data/br_agents/br_agents_4p_1s_8c_m')


    #build_complete_br('data/states/state_4_8.pkl', 'data/br_agents/br_agents_4p_4s_8c_f')
    #agent1.load("data/ppo_agents/ppo_agent_4p_4s_12c_f_perfect")
    #approx_exploitability(50, agent1)
    #approx_exploitability(50, random_agent)







    #env.set_agents(agents)
