import torch
import torch.nn.functional as F
# import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
from Board import Board
from QNetwork import QNetwork
import utils
from ReplayBuffer import ReplayBuffer

# Set all random seeds
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(RANDOM_SEED)

# Set global information
OUR_PLAYER_INDEX = 1
opponent_player_index = 1 if OUR_PLAYER_INDEX == 2 else 2
BOARD_SIZE = 8
HIDDEN_DIM_SIZE = 128
N_STATES = BOARD_SIZE ** 2
N_ACTIONS = BOARD_SIZE ** 2
folder_name = f'{BOARD_SIZE}x{BOARD_SIZE}_target_learner_2/'

# Set number of training iterations and indices for which an intermediate model should be saved
NUM_TRAJECTORIES = 15000
save_iters = [t for t in range(NUM_TRAJECTORIES) if t % 100 == 0 or t == NUM_TRAJECTORIES - 1]

# warmup steps to collect the data first
WARMUP = 1000

# NN-training parameters
gamma = 0.99
EPSILON = 0.1
# EPSILON_DECAY = 0.99975
EPSILON_DECAY = 1./(NUM_TRAJECTORIES - WARMUP)
SOFT_UPDATE = 0.01
BATCH_SIZE = 512


if __name__ == '__main__':
    # Use all available threads for training :D
    thread_count = torch.get_num_threads()
    torch.set_num_threads(thread_count)

    # Generate matrix of action indices (BOARD_SIZExBOARD_SIZE with range 0-BOARD_SIZE**2 values)
    action_board = utils.generate_action_board(board_size=BOARD_SIZE)

    # Initialize all networks
    policy_network = QNetwork(n_states=N_STATES, n_actions=N_ACTIONS, hidden_dim=HIDDEN_DIM_SIZE).to(device)
    policy_optimizer = torch.optim.Adam(params=policy_network.parameters(), lr=5e-4)
    target_network = QNetwork(n_states=N_STATES, n_actions=N_ACTIONS, hidden_dim=HIDDEN_DIM_SIZE).to(device)

    # placeholders for rewards for each episode
    iteration_rewards = []
    policy_losses = []
    # declaring the replay buffer
    transition_buffer = ReplayBuffer(10000, seed=RANDOM_SEED)

    err = False

    # iterating through trajectories
    for tau in tqdm(range(NUM_TRAJECTORIES)):
        # resetting the environment
        # state, info = env.reset(seed=123)
        board = Board(board_size=BOARD_SIZE)

        # setting done to False for while loop
        done = False
        player = 1
        t = -1  # number of steps tracker
        reward = 0
        n_flips = 0  # To catch infinite loops (very rare events)
        while not done:
            # retrieving Q(s, a) for all possible a
            # Start by getting state of the board
            state = utils.get_state(board)
            # Then get legal actions from current state given the player
            legal_actions = utils.get_legal_action_indices(board, player=player)
            t += 1  # Increase step counter

            # Check if there are NOT any legal moves for the current player
            if not legal_actions.any():
                # If no legal moves for player, skip and change players
                n_flips += 1
                # If you have flipped twice, it means infinite loop found, check reward, break
                if n_flips == 2:
                    reward = utils.check_reward(board, board.get_board(), current_player=OUR_PLAYER_INDEX)
                    break
                if player == 1:
                    player = 2
                else:
                    player = 1
                continue

            n_flips = 0  # This means that there was at least one legal move, so no change of players

            # If player == 2 handles the prebuilt 'player's' turn
            if player == opponent_player_index:
                action_q_values = utils.apply_filter(target_network(torch.tensor(state).to(device)), legal_actions)
                action = torch.argmax(action_q_values).detach().cpu().numpy()
            else:
                # Get Q values from our network for all legal moves
                action_q_values = utils.apply_filter(policy_network(torch.tensor(state).to(device)), legal_actions)

                # Action list, best Q action, random action
                a = [torch.argmax(action_q_values).detach().cpu().numpy(),
                     np.random.choice(np.where(legal_actions == 1)[0])]

                if t < 0:  # Early game robust-ness, to increase variety in opening moves
                    action = np.random.choice(a, p=[0.5, 0.5])
                else:
                    # epsilon-greedy action
                    action = np.random.choice(
                        a, p=[1 - EPSILON, EPSILON])

            # keeping track of previous state
            prev_state = state.copy()

            # environment step, make chosen action
            x_val, y_val = np.where(action_board == action)
            state_post_move, num_flip = board.make_move(x_val[0], y_val[0], player=player)

            # Ensure model chose a legal action
            if num_flip <= 0:
                board.print_board()
                print(f'number of flipped pieces for move: {x_val}, {y_val} is 0')
                err = True
                break
            # assert num_flip > 0, 'Number of tiles flipped is 0, this is an illegal move'

            board.set_board(copy.deepcopy(state_post_move))
            state = utils.get_state(board)

            # Check if next state is a terminal node (has no legal actions) for both players
            done = (board.is_terminal_node(player=1, board_state=state_post_move) and
                    board.is_terminal_node(player=2, board_state=state_post_move))

            if done:  # Game is over
                # reward = number of disks player 1 - number of disks player 2
                reward = utils.check_reward(board, state_post_move, current_player=OUR_PLAYER_INDEX)
            else:
                reward = 0

            # Append all information to the transition buffer as done in DQN notebook
            transition_buffer.append((prev_state, action, reward, state, done))

            # Change players
            if player == 1:
                player = 2
            else:
                player = 1

        if err:
            err = False
            continue

        # logging the calculated reward of final step as iteration reward
        iteration_rewards.append(reward)

        # After warm up phase, update Q-values (taken from DQN notebook)
        if len(transition_buffer) > WARMUP:
            states, actions, rewards, next_states, dones = transition_buffer.sample(sample_size=BATCH_SIZE,
                                                                                    device=device)
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
            # Decay Epsilon value
            EPSILON -= EPSILON_DECAY

        # if epoch == a saving iteration, save model to allow for external benchmarking/validation
        if tau in save_iters:
            file_name = f'{BOARD_SIZE}x{BOARD_SIZE}_model_step_{tau}.pth'
            torch.save(policy_network.state_dict(), folder_name + file_name)

    np.savetxt(folder_name + 'iteration_rewards.txt', iteration_rewards)
    # Plot training curve (iteration reward curve)
    plt.figure(figsize=(12, 9))
    plt.plot(utils.running_mean(iteration_rewards, 100))
    plt.title("DQN cumulative rewards")
    plt.grid()

    plt.show()
