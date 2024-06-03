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
BOARD_SIZE = 8
HIDDEN_DIM_SIZE = 128
N_STATES = BOARD_SIZE ** 2
N_ACTIONS = BOARD_SIZE ** 2

# Set number of training iterations and indices for which an intermediate model should be saved
NUM_TRAJECTORIES = 10000
save_iters = [t for t in range(NUM_TRAJECTORIES) if t % 100 == 0 or t == NUM_TRAJECTORIES - 1]

# warmup steps to collect the data first
WARMUP = 1000

# NN-training parameters
gamma = 0.99
EPSILON = 0.1
EPSILON_DECAY = 0.99975
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

    for i in range(NUM_TRAJECTORIES):
        board = Board(board_size=BOARD_SIZE)
        done = False
        while not done:
            pass
            # Observe current state s
            # For all actions a' in s use the NN to compute Q(s,a')
            # Select an action a using a policy pi
            # Q_output <- Q(s, a')
            # Execuse action a
            # Receive an immediate reward r
            # Observe the resulting new state s'
            # For all actions a' in s' use the NN to compute Q(s', a')
            # According to Eq (11) compute Q_target <- Q(s, a)
                # (1-learning rate)Q(s,a) + learning rate (r + \gamma max Q(s', a'))
            # Adjust the NN by backpropagating the error (Q_target - Q_output)
            # s <- s'
