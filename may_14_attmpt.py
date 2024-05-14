import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import deque
import random


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
        return self.linear3(x)


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


policy_network = QNetwork(n_states=4, n_actions=2, hidden_dim=128).to(device)
policy_optimizer = torch.optim.Adam(params=policy_network.parameters(), lr=5e-4)
target_network = QNetwork(n_states=4, n_actions=2, hidden_dim=128).to(device)


def parameter_update(source_model, target_model, tau):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau)*target_param.data)


env = gym.make('CartPole-v1')

NUM_TRAJECTORIES = 2000
MAX_EPISODE_LENGTH = 500

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
    state, info = env.reset(seed=123)

    # TODO: Need to add board.reset() option which returns the board

    # setting done to False for while loop
    done = False
    t = 0
    while done == False and t < MAX_EPISODE_LENGTH:
        # retrieving Q(s, a) for all possible a
        action_q_values = policy_network(torch.tensor(state).to(device))
        # epsilon-greedy action
        # TODO: Think about number of actions are available to our model, what should output side be...???
        action = np.random.choice(
            [torch.argmax(action_q_values.flatten()).detach().cpu().numpy(),
             np.random.choice([0, 1])],
            p=[1 - EPSILON, EPSILON])
        # keeping track of previous state
        prev_state = state
        # environment step
        state, reward, done, truncation, info = env.step(action)
        # TODO: Our 'board.step' equivalent is 'board.make_move()', need to return board state, reward, 'done,' and more

        transition_buffer.append((prev_state, action, reward, state, done))

        t += 1

    # logging the episode length as a cumulative reward
    cumulative_rewards.append(t)

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


plt.figure(figsize=(12,9))
plt.plot(running_mean(cumulative_rewards, 50))
plt.title("DQN cumulative rewards")
plt.grid()


