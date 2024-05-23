import random
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, size, seed):
        self.memory = deque([], maxlen=size)
        random.seed(seed)

    # this method samples transitions and returns tensors of each type registered in the environment step
    def sample(self, sample_size, device):
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
