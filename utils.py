import numpy as np
import torch
from QNetwork import QNetwork

action_board = None  # Create access to the action board for other functions


# running mean function for the purpose of visualization
def running_mean(x, N):
    """
    Running mean function for the purpose of visualization
    Arguments:
        x (list): The list of objects to take the window-based mean of
        N (int): The size of the sliding mean window
    Return:
         np.ndarray of means
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def parameter_update(source_model, target_model, tau):
    """
    From target model to source model, update parameters
    Arguments:
        source_model (QNetwork): source model
        target_model (QNetwork): target model
        tau (float): Update value (Soft update)
    """
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def generate_action_board(board_size):
    """
    Generates a np.ndarray of action indices
    Arguments:
        board_size (int): board size
    Return:
        np.ndarray(board_size, board_size) range from 0-board_size**2 reshaped to be board_size x board_size
    """
    global action_board
    action_board = np.arange(0, board_size ** 2, 1).reshape((board_size, board_size))
    return action_board


def get_state(board):
    """
    Gets the current state of the board
    Arguments:
        board (Board.Board): the current Othello board with get_board() function
    Return:
        flat np.array of the board state (1's for player 1, 2's for player 2, 0 for empty space)
    """
    np_board = board.board_to_numpy(board.get_board())
    return np_board.flatten()


def get_legal_action_indices(board, player):
    """
    Creates an array of 0's and 1's with 1's corresponding to legal actions for the given player
    Arguments:
        board (Board.Board): board with current state of play
        player (int): player index
    Return:
        binary np.array of legal actions
    """
    return get_legal_actions(board.board_size, actions=board.get_sorted_nodes(player))


def get_legal_actions(board_size, actions):
    """
    Creates an array of 0's and 1's where legal actions exist
    Arguments:
        board_size (int): board size
        actions (list): list of legal actions
    Return:
        binary np.array of legal actions
    """
    vals = np.zeros(board_size ** 2, dtype=int)
    for x, y in actions:
        vals[action_board[x, y]] = 1
    return vals


def apply_filter(output, mask):
    """
    Given an array and a mask, turn all values of array to 0 outside of mask
    Arguments:
        output (array): array to be masked
        mask (array): binary mask array
    Return:
        masked output
    """
    output[mask == 0] = -np.infty
    return output


def check_reward(board, board_state, current_player):
    """
    Gets the total reward from the current board and state of play
    Arguments:
        board (Board.Board): Othello board object
        board_state (np.array): current state of play as given by utils.get_state()
    Return:
        int points player 1 - points player 2
    """
    opposite_player = 1 if current_player == 2 else 2
    current_player_discs = board.count_board(board_state, player=current_player)  # Points player 1
    opposite_player_discs = board.count_board(board_state, player=opposite_player)  # Points player 2
    return current_player_discs - opposite_player_discs


def get_network_attributes(state_dict):
    """
    Get list of attributes from state dict to determine size of Neural Net from state dictionary
    Arguments:
        state_dict (dict): torch produced state dictionary containing NN information
    Return:
        tuple of #states, #hidden dim, #outputs
    """
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
    """
    from a file, load a QNetwork to torch device
    Arguments:
        file_name (string): file name of QNetwork state dictionary
        device (torch.device): torch device (GPU if you have one...)
    Return:
        QNetwork
    """
    state_dict = torch.load(file_name)
    n_states, hidden_dim_size, n_actions = get_network_attributes(state_dict)

    model = QNetwork(n_states=n_states, n_actions=n_actions, hidden_dim=hidden_dim_size).to(device)
    model.load_state_dict(torch.load(file_name))
    return model
