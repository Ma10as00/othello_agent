# Reversi/Othello Board Game using Minimax and Alpha-Beta Pruning
# https://en.wikipedia.org/wiki/Reversi
# https://en.wikipedia.org/wiki/Computer_Othello
# https://en.wikipedia.org/wiki/Minimax
# https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
# https://en.wikipedia.org/wiki/Negamax
# https://en.wikipedia.org/wiki/Principal_variation_search
# FB36 - 20160831
import os
import copy
import numpy as np


class Board:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = []
        self.dirx = [-1, 0, 1, -1, 1, -1, 0, 1]
        self.diry = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.empty_char = '-'
        self.player_chars = ['X', 'O']
        self.position_value_matrix = np.array([[100, -20, 10,  5,  5, 10, -20, 100],
                                               [-20, -50, -2, -2, -2, -2, -50, -20],
                                               [ 10,  -2, -1, -1, -1, -1,  -2,  10],
                                               [  5,  -2, -1, -1, -1, -1,  -2,   5],
                                               [  5,  -2, -1, -1, -1, -1,  -2,   5],
                                               [ 10,  -2, -1, -1, -1, -1,  -2,  10],
                                               [-20, -50, -2, -2, -2, -2, -50, -20],
                                               [100, -20, 10,  5,  5, 10, -20, 100]])
        self.minEvalBoard = np.sum(self.position_value_matrix[self.position_value_matrix < 0]) - 1
        self.maxEvalBoard = np.sum(self.position_value_matrix[self.position_value_matrix > 0]) + 1
        self.init_board()

    def init_board(self):
        n = self.board_size
        self.board = [[self.empty_char for _ in range(n)] for _ in range(n)]
        if n % 2 == 0:
            z = int((n - 2) / 2)
            self.board[z][z] = self.player_chars[0]
            self.board[z + 1][z] = self.player_chars[1]
            self.board[z][z + 1] = self.player_chars[1]
            self.board[z + 1][z + 1] = self.player_chars[0]

    def set_board(self, bd):
        self.board = bd

    def print_board(self):
        n = self.board_size
        m = len(str(n - 1))
        for y in range(n):
            row = ''
            for x in range(n):
                row += self.get_board()[y][x]
                row += ' ' * m
            print(row + ' ' + str(y))
        # print
        row = ''
        for x in range(n):
            row += str(x).zfill(m) + ' '
        print(row + '\n')

    def make_move(self, x, y, player):
        totctr = 0  # total number of opponent pieces taken
        n = self.board_size
        temp_board = copy.deepcopy(self.get_board())
        temp_board[y][x] = self.player_chars[player - 1]
        for dxx, dyy in zip(self.dirx, self.diry):  # 8 directions
            ctr = 0
            for i in range(n):
                dx = x + dxx * (i + 1)
                dy = y + dyy * (i + 1)
                if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                    ctr = 0
                    break
                elif temp_board[dy][dx] == self.player_chars[player - 1]:
                    break
                elif temp_board[dy][dx] == self.empty_char:
                    ctr = 0
                    break
                else:
                    ctr += 1
            for i in range(ctr):
                dx = x + dxx * (i + 1)
                dy = y + dyy * (i + 1)
                temp_board[dy][dx] = self.player_chars[player - 1]
            totctr += ctr
        return temp_board, totctr

    def valid_move(self, x, y, player):
        n = self.board_size
        if x < 0 or x > n - 1 or y < 0 or y > n - 1:
            return False
        if self.get_board()[y][x] != self.empty_char:
            return False
        boardTemp, totctr = self.make_move(x, y, player)
        if totctr == 0:
            return False
        return True

    def our_EvalBoard(self, b, player):
        tot = 0
        n = self.board_size
        for row in range(n):
            for col in range(n):
                if b[row][col] == self.player_chars[player - 1]:
                    tot += self.position_value_matrix[row][col]
        return tot

    # if no valid move(s) possible then True
    def is_terminal_node(self, player):
        n = self.board_size
        for y in range(n):
            for x in range(n):
                if self.valid_move(x, y, player):
                    return False
        return True

    def get_sorted_nodes(self, player):
        n = self.board_size
        sortedNodes = []
        for y in range(n):
            for x in range(n):
                if self.valid_move(x, y, player):
                    boardTemp, totctr = self.make_move(x, y, player)
                    sortedNodes.append((boardTemp, self.our_EvalBoard(boardTemp, player)))
        sortedNodes = sorted(sortedNodes, key=lambda node: node[1], reverse=True)
        sortedNodes = [node[0] for node in sortedNodes]
        return sortedNodes

    def get_board(self):
        return self.board


# def EvalBoard(board, player):
#     tot = 0
#     for y in range(n):
#         for x in range(n):
#             if board[y][x] == player:
#                 if (x == 0 or x == n - 1) and (y == 0 or y == n - 1):
#                     tot += 4  # corner
#                 elif (x == 0 or x == n - 1) or (y == 0 or y == n - 1):
#                     tot += 2  # side
#                 else:
#                     tot += 1
#     return tot


def Minimax(board_state, player, depth, maximizingPlayer):
    board = Board()
    board.set_board(board_state)
    if depth == 0 or board.is_terminal_node(player):
        return board.our_EvalBoard(board.get_board(), player)
    if maximizingPlayer:
        best_value = board.minEvalBoard
    else:
        best_value = board.maxEvalBoard
    n = board.board_size
    for y in range(n):
        for x in range(n):
            if board.valid_move(x, y, player):
                boardTemp, totctr = board.make_move(x, y, player)
                if maximizingPlayer:
                    v = Minimax(boardTemp, player, depth - 1, False)
                    best_value = max(best_value, v)
                else:
                    v = Minimax(boardTemp, player, depth - 1, True)
                    best_value = min(best_value, v)
    return best_value


def AlphaBeta(board_state, player, depth, alpha, beta, maximizingPlayer):
    board = Board()
    board.set_board(board_state)
    if depth == 0 or board.is_terminal_node(player):
        return board.our_EvalBoard(board.get_board(), player)

    if maximizingPlayer:
        v = board.minEvalBoard
    else:
        v = board.maxEvalBoard
    n = board.board_size
    for y in range(n):
        for x in range(n):
            if board.valid_move(x, y, player):
                boardTemp, totctr = board.make_move(x, y, player)
                if maximizingPlayer:
                    v = max(v, AlphaBeta(boardTemp, player, depth - 1, alpha, beta, False))
                    alpha = max(alpha, v)
                else:
                    v = min(v, AlphaBeta(boardTemp, player, depth - 1, alpha, beta, True))
                    beta = min(beta, v)
                if beta <= alpha:
                    break  # beta cut-off
        return v


def AlphaBetaSN(board_state, player, depth, alpha, beta, maximizingPlayer):
    board = Board()
    board.set_board(board_state)
    if depth == 0 or board.is_terminal_node(player):
        return board.our_EvalBoard(board.get_board(), player)
    sortedNodes = board.get_sorted_nodes(player)
    if maximizingPlayer:
        v = board.minEvalBoard
    else:
        v = board.maxEvalBoard
    for boardTemp in sortedNodes:
        if maximizingPlayer:
            v = max(v, AlphaBetaSN(boardTemp, player, depth - 1, alpha, beta, False))
            alpha = max(alpha, v)
        else:
            v = min(v, AlphaBetaSN(boardTemp, player, depth - 1, alpha, beta, True))
            beta = min(beta, v)
        if beta <= alpha:
            break  # beta cut-off
    return v


def Negamax(board_state, player, depth, color):
    board = Board()
    board.set_board(board_state)
    if depth == 0 or board.is_terminal_node(player):
        return color * board.our_EvalBoard(board.get_board(), player)
    best_value = board.minEvalBoard
    n = board.board_size
    for y in range(n):
        for x in range(n):
            if board.valid_move(x, y, player):
                boardTemp, totctr = board.make_move(x, y, player)
                v = -Negamax(boardTemp, player, depth - 1, -color)
                best_value = max(best_value, v)
    return best_value


def NegamaxAB(board_state, player, depth, alpha, beta, color):
    board = Board()
    board.set_board(board_state)
    if depth == 0 or board.is_terminal_node(player):
        return color * board.our_EvalBoard(board.get_board(), player)
    best_value = board.minEvalBoard
    n = board.board_size
    for y in range(n):
        for x in range(n):
            if board.valid_move(x, y, player):
                boardTemp, totctr = board.make_move(x, y, player)
                v = -NegamaxAB(boardTemp, player, depth - 1, -beta, -alpha, -color)
                best_value = max(best_value, v)
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
    return best_value


def NegamaxABSN(board_state, player, depth, alpha, beta, color):
    board = Board()
    board.set_board(board_state)
    if depth == 0 or board.is_terminal_node(player):
        return color * board.our_EvalBoard(board.get_board(), player)
    sorted_nodes = board.get_sorted_nodes(player)
    best_value = board.minEvalBoard
    for board_temp in sorted_nodes:
        v = -NegamaxABSN(board_temp, player, depth - 1, -beta, -alpha, -color)
        best_value = max(best_value, v)
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best_value


def Negascout(board_state, player, depth, alpha, beta, color):
    board = Board()
    board.set_board(board_state)
    if depth == 0 or board.is_terminal_node(player):
        return color * board.our_EvalBoard(board.get_board(), player)
    first_child = True
    n = board.board_size
    for y in range(n):
        for x in range(n):
            if board.valid_move(x, y, player):
                board_temp, totctr = board.make_move(x, y, player)
                if not first_child:
                    score = -Negascout(board_temp, player, depth - 1, -alpha - 1, -alpha, -color)
                    if alpha < score < beta:
                        score = -Negascout(board_temp, player, depth - 1, -beta, -score, -color)
                else:
                    first_child = False
                    score = -Negascout(board_temp, player, depth - 1, -beta, -alpha, -color)
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
    return alpha


def NegascoutSN(board_state, player, depth, alpha, beta, color):
    board = Board()
    board.set_board(board_state)
    if depth == 0 or board.is_terminal_node(player):
        return color * board.our_EvalBoard(board.get_board(), player)
    sorted_nodes = board.get_sorted_nodes(player)
    first_child = True
    for board_temp in sorted_nodes:
        if not first_child:
            score = -NegascoutSN(board_temp, player, depth - 1, -alpha - 1, -alpha, -color)
            if alpha < score < beta:
                score = -NegascoutSN(board_temp, player, depth - 1, -beta, -score, -color)
        else:
            first_child = False
            score = -NegascoutSN(board_temp, player, depth - 1, -beta, -alpha, -color)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return alpha


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
