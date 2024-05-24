import numpy as np
import torch
from QNetwork import QNetwork


action_board = None


# running mean function for the purpose of visualization
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def parameter_update(source_model, target_model, tau):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def generate_action_board(board_size):
    global action_board
    action_board = np.arange(0, board_size ** 2, 1).reshape((board_size, board_size))
    return action_board


def get_state(board):
    np_board = board.board_to_numpy(board.get_board())
    return np_board.flatten()


def get_legal_actions(board, player):#, action_board):
    return get_legal_action_indices(board.board_size, actions=board.get_sorted_nodes(player))#, action_board)


def get_legal_action_indices(board_size, actions):#, action_board):
    vals = np.zeros(board_size ** 2, dtype=int)
    for x, y in actions:
        vals[action_board[x, y]] = 1
    return vals


def apply_filter(output, legal_actions):
    output[legal_actions == 0] = 0
    return output


def check_reward(board, state):
    p1 = board.count_board(state, player=1)
    p2 = board.count_board(state, player=2)
    return p1 - p2  # return # discs player 1 (our agent) has - # discs player 2 has


def get_network_attributes(state_dict):
    attribute_list = []
    num_items = len(state_dict.items())
    for i, item in enumerate(state_dict.items()):
        if i % 2 == 0:
            if i == num_items - 2:
                attribute_list.append(item[1].shape[0])
            else:
                attribute_list.append(item[1].shape[1])

    return tuple(attribute_list)


def load_trained_network(file_name, device):
    state_dict = torch.load(file_name)
    n_states, hidden_dim_size, n_actions = get_network_attributes(state_dict)

    model = QNetwork(n_states=n_states, n_actions=n_actions, hidden_dim=hidden_dim_size).to(device)
    model.load_state_dict(torch.load(file_name))
    return model
