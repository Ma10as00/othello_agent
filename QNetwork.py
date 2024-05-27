import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    QNetwork which extends the torch.nn.Module class
    """
    def __init__(self, n_states, n_actions, hidden_dim):
        """
        Initialized nn.Module with one hidden layer
        Arguments:
            n_states (int): number of states for input layer
            n_actions (int): number of output features
            hidden_dim (int): number of nodes for the hidden dimension
        """
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        """
        Forward pass of Neural Network
        Layer 1: Rectified Linear Unit
        Layer 2: Rectified Linear Unit
        Layer 3: sigmoid function of above outputs
        Arguments:
            state (np.array): current state of the board (0 for empty space, 1 for player 1, 2 for player 2)
        Return:
            output: sigmoid of linear3 output nodes
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return F.sigmoid(self.linear3(x))