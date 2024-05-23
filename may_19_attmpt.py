import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import deque
import random
import copy
from Board import Board


torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(123)


# running mean function for the purpose of visualization
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return F.sigmoid(self.linear3(x))


class ReplayBuffer:
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    # this method samples transitions and returns tensors of each type registered in the environment step
    def sample(self, sample_size):
        sample = random.sample(self.memory, sample_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for x in sample:
            states.append(x[0])
            actions.append(x[1])
            rewards.append(x[2])
            next_states.append(x[3])
            dones.append(x[4])
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.tensor(next_states).to(device)
        dones = torch.tensor(dones, dtype=torch.int).to(device)
        return states, actions, rewards, next_states, dones

    # add transition to the buffer
    def append(self, item):
        self.memory.append(item)

    def __len__(self):
        return len(self.memory)


board_size = 4
board = Board(board_size=board_size)
# print(board.get_board())
# print(board.board_to_numpy(board.get_board()))
# print(board.our_EvalBoard(board.get_board(), 2))
# print(board.count_board(board.get_board(), 2))

n_states = n_actions = board_size ** 2

action_board = np.arange(0, n_actions, 1).reshape(board.board_to_numpy().shape)
# print(action_board)


def get_state(board):
    np_board = board.board_to_numpy(board.get_board())
    return np_board.flatten()


def get_legal_actions(board, player):
    actions = board.get_sorted_nodes(player)
    return get_legal_action_indices(actions)


def get_legal_action_indices(actions):
    vals = np.zeros(n_actions, dtype=int)
    for x, y in actions:
        vals[action_board[x, y]] = 1
    return vals


def apply_filter(output, legal_actions):
    output[np.where(legal_actions == 0)[0]] = 0
    return output


def check_reward(board, state):
    p1 = board.count_board(state, player=1)
    p2 = board.count_board(state, player=2)
    # if p1 > p2:
    #     return 1
    # elif p2 > p1:
    #     return -1
    # else:
    #     return 0
    return p1 - p2  # return # discs player 1 (our agent) has - # discs player 2 has


actions = board.get_sorted_nodes(1)
# print(get_legal_action_indices(actions))
# print(get_legal_action_indices(actions).reshape(action_board.shape))
# exit(0)


policy_network = QNetwork(n_states=n_states, n_actions=n_actions, hidden_dim=128).to(device)
policy_optimizer = torch.optim.Adam(params=policy_network.parameters(), lr=5e-4)
target_network = QNetwork(n_states=n_states, n_actions=n_actions, hidden_dim=128).to(device)


def parameter_update(source_model, target_model, tau):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# env = gym.make('CartPole-v1')
# env = othello board play yay

# NUM_TRAJECTORIES = 2000
MAX_EPISODE_LENGTH = 4 * n_states
NUM_TRAJECTORIES = 2000

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
transition_buffer = ReplayBuffer(10000)
# iterating through trajectories


for tau in tqdm(range(NUM_TRAJECTORIES)):
    # resetting the environment
    # state, info = env.reset(seed=123)
    board = Board(board_size=board_size)

    # setting done to False for while loop
    done = False
    t = 0
    player = 1
    reward = 0
    while not done and t < MAX_EPISODE_LENGTH:
        # retrieving Q(s, a) for all possible a
        state = get_state(board)
        # print(state)
        legal_actions = get_legal_actions(board, player=player)

        if len(np.where(legal_actions == 1)[0]) == 0:
            # If no legal moves for player, skip and change players
            t += 1
            if player == 1:
                player = 2
            else:
                player = 1
            continue

        action_q_values = apply_filter(policy_network(torch.tensor(state).to(device)), legal_actions)
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
        state = get_state(board)
        # print(state)
        #
        # print(board.is_terminal_node(player=1, board=state_post_move))
        done = (board.is_terminal_node(player=1, board=state_post_move) and
                board.is_terminal_node(player=2, board=state_post_move))

        if done:
            # reward = 1 if player 1 has more discs, 0 if =, -1 if fewer
            # reward = 100 * check_reward(board, state_post_move)
            reward = check_reward(board, state_post_move)
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
        states, actions, rewards, next_states, dones = transition_buffer.sample(sample_size=BATCH_SIZE)
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
        parameter_update(policy_network, target_network, SOFT_UPDATE)


file_name = f'{board_size}x{board_size}_model.pth'
# Save Learned Policy Network
torch.save(policy_network.state_dict(), file_name)


plt.figure(figsize=(12, 9))
plt.plot(running_mean(cumulative_rewards, 100))
plt.title("DQN cumulative rewards")
plt.grid()

plt.show()