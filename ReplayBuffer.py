import random
import torch
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Class taken from DQN Notebook.
    Creates a list of memory up to a given size which is sampled from during learning steps
    """
    def __init__(self, size, seed):
        """
        Initialize the buffer's memory and define maximum size
        Arguments:
            size (int): length of buffer
            seed (int): random seed
        """
        self.memory = deque([], maxlen=size)
        random.seed(seed)

    # this method samples transitions and returns tensors of each type registered in the environment step
    def sample(self, sample_size, device):
        """
        Samples transitions and returns tensors of each type registered in the environment step
        Arguments:
            sample_size (int): batch size of sampling
            device (torch.device): torch device
        Return:
            states, actions, rewards, next_states, dones: torch tensors
        """
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

        # Convert lists to tensors
        states = torch.tensor(np.array(states)).to(device)
        actions = torch.tensor(np.array(actions)).to(device)
        rewards = torch.tensor(np.array(rewards)).to(device)
        next_states = torch.tensor(np.array(next_states)).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.int).to(device)
        return states, actions, rewards, next_states, dones

    # add transition to the buffer
    def append(self, item):
        """
        append an item to the buffer's memory
        Arguments:
            item: item to be added
        Return:
            None
        """
        self.memory.append(item)

    def __len__(self):
        """
        Return:
            (int) length of replay buffer's memory
        """
        return len(self.memory)
