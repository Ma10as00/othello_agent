from Environment.players import *
from Environment.Board import Board
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(123)


def BestMove(board_state, player, move):
    board = Board()
    board.set_board(board_state)
    max_points = -np.infty
    points = -np.infty
    mx = -1
    my = -1
    if opt == 10:
        return our_self_learned_player(board, player, move)
    elif opt == 11:
        return our_positional_learned_player(board, player, move)
    n = board.board_size
    for row in range(n):
        for col in range(n):
            if board.valid_move(row, col, player):
                board_temp, totctr = board.make_move(row, col, player)
                if opt == 0:
                    points = board.their_eval_board(board_temp, player)
                elif opt == 1:
                    points = Minimax(board_temp, player, depth, True)
                elif opt == 2:
                    points = AlphaBeta(board_temp, player, depth,
                                       board.minEvalBoard, board.maxEvalBoard, True)
                elif opt == 3:
                    points = Negamax(board_temp, player, depth, 1)
                elif opt == 4:
                    points = NegamaxAB(board_temp, player, depth,
                                       board.minEvalBoard, board.maxEvalBoard, 1)
                elif opt == 5:
                    points = Negascout(board_temp, player, depth,
                                       board.minEvalBoard, board.maxEvalBoard, 1)
                elif opt == 6:
                    points = AlphaBetaSN(board_temp, player, depth,
                                         board.minEvalBoard, board.maxEvalBoard, True)
                elif opt == 7:
                    points = NegamaxABSN(board_temp, player, depth,
                                         board.minEvalBoard, board.maxEvalBoard, 1)
                elif opt == 8:
                    points = NegascoutSN(board_temp, player, depth,
                                         board.minEvalBoard, board.maxEvalBoard, 1)
                elif opt == 9:
                    points = board.our_EvalBoard(board_temp, player, board.position_value_matrix)
                if points > max_points:
                    max_points = points
                    mx = row
                    my = col
    return mx, my


print('REVERSI/OTHELLO BOARD GAME')
print('0: EvalBoard')
print('1: Minimax')
print('2: Minimax w/ Alpha-Beta Pruning')
print('3: Negamax')
print('4: Negamax w/ Alpha-Beta Pruning')
print('5: Negascout (Principal Variation Search)')
print('6: Minimax w/ Alpha-Beta Pruning w/ Sorted Nodes')
print('7: Negamax w/ Alpha-Beta Pruning w/ Sorted Nodes')
print('8: Negascout (Principal Variation Search) w/ Sorted Nodes')
print('9: Our EvalBoard with position value matrix from van Eck & van Wezel')
print('10: Our Learned Agent - self play')
print('11: Our Learned Agent - positional play')
opt = int(input('Select AI Algorithm: '))
if 9 > opt > 0:
    depth = 4
    depthStr = input('Select Search Depth (DEFAULT: 4): ')
    if depthStr != '':
        depth = int(depth)
PLAYER = int(input('Select which position you would like to play: 1 or 2'))
OPPONENT = 1 if PLAYER == 2 else 2

print('\n1: User 2: AI (Just press \'Enter\' or \'q\' for Exit!)')

board = Board()

move = 0
iteration = True
while iteration:
    for p in range(2):
        board.print_board()
        player = p + 1
        print(f'PLAYER: {player} as {board.player_chars[player - 1]}')
        if (board.is_terminal_node(1, copy.deepcopy(board.get_board()))
                and board.is_terminal_node(2, copy.deepcopy(board.get_board()))):
            our_score = board.count_board(board.get_board(), player=PLAYER)
            ai_score = board.count_board(board.get_board(), player=OPPONENT)
            # print('Player cannot play! Game ended!')
            print(f'Score User: {our_score}')
            print(f'Score AI  : {ai_score}\n')
            iteration = False
            break
        if board.is_terminal_node(player, copy.deepcopy(board.get_board())):
            continue

        if player == PLAYER:  # user's turn
            while True:
                row_col = input('ROW COL: ')
                if row_col == '' or row_col == 'q':
                    exit(0)
                (row, col) = row_col.split()
                row = int(row)
                col = int(col)
                if board.valid_move(row, col, player):
                    temp_board, totctr = board.make_move(row, col, player)
                    print('# of pieces taken: ' + str(totctr))
                    board.set_board(temp_board)
                    break
                else:
                    print('Invalid move! Try again!')
        else:  # AI's turn
            row, col = BestMove(board.get_board(), player, move)
            if not (row == -1 and col == -1):
                temp_board, totctr = board.make_move(row, col, player)
                print('AI played (ROW COL): ' + str(row) + ' ' + str(col))
                print('# of pieces taken: ' + str(totctr))
                board.set_board(temp_board)
    move += 1
