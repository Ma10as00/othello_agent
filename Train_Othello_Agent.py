import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
from Board import Board
from QNetwork import QNetwork
import utils
from ReplayBuffer import ReplayBuffer

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(RANDOM_SEED)

BOARD_SIZE = 8
HIDDEN_DIM_SIZE = 128
N_STATES = BOARD_SIZE ** 2
N_ACTIONS = BOARD_SIZE ** 2


if __name__ == '__main__':
    board = Board(board_size=BOARD_SIZE)
    action_board = utils.generate_action_board(board_size=BOARD_SIZE)
    # print(board.get_board())
    # print(board.board_to_numpy(board.get_board()))
    # print(board.our_EvalBoard(board.get_board(), 2))
    # print(board.count_board(board.get_board(), 2))

    # n_states = n_actions = board_size ** 2

    # actions = board.get_sorted_nodes(1)
    # print(get_legal_action_indices(actions))
    # print(get_legal_action_indices(actions).reshape(action_board.shape))
    # exit(0)

    policy_network = QNetwork(n_states=N_STATES, n_actions=N_ACTIONS, hidden_dim=HIDDEN_DIM_SIZE).to(device)
    policy_optimizer = torch.optim.Adam(params=policy_network.parameters(), lr=5e-4)
    target_network = QNetwork(n_states=N_STATES, n_actions=N_ACTIONS, hidden_dim=HIDDEN_DIM_SIZE).to(device)

    # env = gym.make('CartPole-v1')
    # env = othello board play yay

    # NUM_TRAJECTORIES = 2000
    MAX_EPISODE_LENGTH = 4 * N_STATES
    NUM_TRAJECTORIES = 14000

    gamma = 0.99
    EPSILON = 0.05
    SOFT_UPDATE = 0.01
    BATCH_SIZE = 512
    # warmup steps to collect the data first
    WARMUP = 1000

    # placeholders for rewards for each episode
    cumulative_rewards = []
    policy_losses = []
    # declaring the replay buffer
    transition_buffer = ReplayBuffer(10000, seed=RANDOM_SEED)
    # iterating through trajectories


    for tau in tqdm(range(NUM_TRAJECTORIES)):
        # resetting the environment
        # state, info = env.reset(seed=123)
        board = Board(board_size=BOARD_SIZE)

        # setting done to False for while loop
        done = False
        t = 0
        player = 1
        reward = 0
        while not done and t < MAX_EPISODE_LENGTH:
            # retrieving Q(s, a) for all possible a
            state = utils.get_state(board)
            # print(state)
            legal_actions = utils.get_legal_actions(board, player=player, action_board=action_board)

            if np.sum(legal_actions) == 0:
                # If no legal moves for player, skip and change players
                t += 1
                if player == 1:
                    player = 2
                else:
                    player = 1
                continue

            action_q_values = utils.apply_filter(policy_network(torch.tensor(state).to(device)), legal_actions)
            # print(np.where(legal_actions == 1)[0])
            # print(action_q_values)

            # TODO: Run through legal actions and find Q function for opponent

            a = [torch.argmax(action_q_values).detach().cpu().numpy(),
                 np.random.choice(np.where(legal_actions == 1)[0])]
            # epsilon-greedy action
            action = np.random.choice(
                a,  # [0, 1],
                p=[1 - EPSILON, EPSILON])
            # if action:
            #     print('I am dead')
            # else:
            #     print('I am alive')
            # action = a[action]
            # print(action)
            # keeping track of previous state
            prev_state = state.copy()
            # environment step
            x_val, y_val = np.where(action_board == action)
            state_post_move, num_flip = board.make_move(x_val[0], y_val[0], player=player)
            assert num_flip > 0, 'Number of tiles flipped is 0, this is an illegal move'
            board.set_board(copy.deepcopy(state_post_move))
            state = utils.get_state(board)
            # print(state)
            #
            # print(board.is_terminal_node(player=1, board=state_post_move))
            done = (board.is_terminal_node(player=1, board=state_post_move) and
                    board.is_terminal_node(player=2, board=state_post_move))

            if done:
                # reward = 1 if player 1 has more discs, 0 if =, -1 if fewer
                # reward = 100 * check_reward(board, state_post_move)
                reward = utils.check_reward(board, state_post_move)
                if reward < 0:
                    reward *= 10
                else:
                    reward *= 5
            else:
                reward = 0
                # reward = num_flip

            # state, reward, done, truncation, info = env.step(action)

            transition_buffer.append((prev_state, action, reward, state, done))
            # print(state.copy().reshape((4, 4)))

            t += 1
            if player == 1:
                player = 2
            else:
                player = 1

        # logging the episode length as a cumulative reward
        # cumulative_rewards.append(t)
        cumulative_rewards.append(reward)

        if len(transition_buffer) > WARMUP:
            states, actions, rewards, next_states, dones = transition_buffer.sample(sample_size=BATCH_SIZE, device=device)
            # getting the maximizing Q-value
            # max(x) return first x values ordered in a decreasing order
            q_target = target_network(torch.tensor(next_states).to(device)).detach().max(1)[0]
            # using Q-values of target network only for non-terminal state
            expected_values = rewards + gamma * q_target * (torch.ones(BATCH_SIZE).to(device) - dones)
            # selecting Q-values of actions taken, using current policy network
            # gather() takes only values indicated by a given index, in this case, action taken
            output = policy_network(states).gather(1, actions.view(-1, 1))
            # computing the loss between r + γ * max Q(s',a) and Q(s,a)
            loss = F.mse_loss(output.flatten(), expected_values)
            policy_losses.append(loss.item())
            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()
            # soft parameter update
            utils.parameter_update(policy_network, target_network, SOFT_UPDATE)


    file_name = f'{BOARD_SIZE}x{BOARD_SIZE}_model.pth'
    # Save Learned Policy Network
    torch.save(policy_network.state_dict(), file_name)

    # model = QNetwork(n_states=n_states, n_actions=n_actions, hidden_dim=128)
    # model.load_state_dict(torch.load(file_name))
    #
    # print(model.state_dict())
    #
    # board = Board(board_size=board_size)
    # state = get_state(board)
    #
    # # model(state)
    #
    # legal_actions = get_legal_actions(board, player=1)
    #
    # action_q_values = apply_filter(model(torch.tensor(state).to(device)), legal_actions)
    #
    # action = torch.argmax(action_q_values).detach().cpu().numpy()
    #
    # x_val, y_val = np.where(action_board == action)
    # state_post_move, num_flip = board.make_move(x_val[0], y_val[0], player=1)
    #
    # print(num_flip, state_post_move)

    plt.figure(figsize=(12, 9))
    plt.plot(utils.running_mean(cumulative_rewards, 100))
    plt.title("DQN cumulative rewards")
    plt.grid()

    plt.show()
