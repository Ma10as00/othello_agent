import copy
import numpy as np
import torch
from Board import Board
import utils


board_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_name = f'{board_size}x{board_size}_model.pth'
trained_model = utils.load_trained_network(file_name, device)


stuff = [[1, 2], [3, 4]]
new_stuff = stuff.copy()

print(stuff, new_stuff)

new_stuff[0][1] = 5

print(stuff, new_stuff)
# board = Board(board_size=board_size)
# state = utils.get_state(board)
# action_board = utils.generate_action_board(board_size)
#
# legal_actions = utils.get_legal_actions(board, player=1, action_board=action_board)
#
# action_q_values = utils.apply_filter(trained_model(torch.tensor(state).to(device)), legal_actions)
#
# action = torch.argmax(action_q_values).detach().cpu().numpy()
#
# x_val, y_val = np.where(action_board == action)
# state_post_move, num_flip = board.make_move(x_val[0], y_val[0], player=1)
#
# board.set_board(copy.deepcopy(state_post_move))
#
# print(num_flip)
#
# board.print_board()
