from players import *
from Board import Board


def BestMove(board_state, player):
    board = Board()
    board.set_board(board_state)
    max_points = board.minEvalBoard
    points = board.minEvalBoard
    mx = -1
    my = -1
    n = board.board_size
    for y in range(n):
        for x in range(n):
            if board.valid_move(x, y, player):
                board_temp, totctr = board.make_move(x, y, player)
                if opt == 0:
                    points = board.our_EvalBoard(board_temp, player)
                elif opt == 1:
                    points = Minimax(board_temp, player, depth, True)
                elif opt == 2:
                    points = AlphaBeta(board, player, depth, board.minEvalBoard, board.maxEvalBoard, True)
                elif opt == 3:
                    points = Negamax(board_temp, player, depth, 1)
                elif opt == 4:
                    points = NegamaxAB(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, 1)
                elif opt == 5:
                    points = Negascout(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, 1)
                elif opt == 6:
                    points = AlphaBetaSN(board, player, depth, board.minEvalBoard, board.maxEvalBoard, True)
                elif opt == 7:
                    points = NegamaxABSN(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, 1)
                elif opt == 8:
                    points = NegascoutSN(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, 1)
                elif opt == 9:
                    points = board.our_EvalBoard(board_temp, player)
                if points > max_points:
                    max_points = points
                    mx = x
                    my = y
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
opt = int(input('Select AI Algorithm: '))
if 9 > opt > 0:
    depth = 4
    depthStr = input('Select Search Depth (DEFAULT: 4): ')
    if depthStr != '':
        depth = int(depth)
print('\n1: User 2: AI (Just press Enter for Exit!)')
board = Board()
while True:
    for p in range(2):
        board.print_board()
        player = p + 1
        print(f'PLAYER: {player} as {board.player_chars[player - 1]}')
        if board.is_terminal_node(player):
            print('Player cannot play! Game ended!')
            print('Score User: ' + str(board.our_EvalBoard(board.get_board(), 1)))
            print('Score AI  : ' + str(board.our_EvalBoard(board.get_board(), 2)))
            exit(0)
        if player == 1:  # user's turn
            while True:
                xy = input('X Y: ')
                if xy == '':
                    exit(0)
                (x, y) = xy.split()
                x = int(x)
                y = int(y)
                if board.valid_move(x, y, player):
                    temp_board, totctr = board.make_move(x, y, player)
                    print('# of pieces taken: ' + str(totctr))
                    board.set_board(temp_board)
                    break
                else:
                    print('Invalid move! Try again!')
        else:  # AI's turn
            x, y = BestMove(board.get_board(), player)
            if not (x == -1 and y == -1):
                temp_board, totctr = board.make_move(x, y, player)
                print('AI played (X Y): ' + str(x) + ' ' + str(y))
                print('# of pieces taken: ' + str(totctr))
                board.set_board(temp_board)
