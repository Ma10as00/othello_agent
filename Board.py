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

    def board_to_numpy(self, board):
        n = len(board)
        np_board = np.zeros((n, n), dtype=np.int8)
        for x in range(n):
            for y in range(n):
                if board[x][y] == self.player_chars[0]:
                    np_board[x][y] = 1
                if board[x][y] == self.player_chars[1]:
                    np_board[x][y] = 2
        return np_board

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

    def make_move(self, x, y, player, board=None):
        totctr = 0  # total number of opponent pieces taken
        n = self.board_size
        if board is not None:
            temp_board = board
        else:
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

    def valid_move(self, x, y, player, board=None):
        n = self.board_size
        if x < 0 or x > n - 1 or y < 0 or y > n - 1:
            return False
        if board is not None:
            if board[y][x] != self.empty_char:
                return False
            boardTemp, totctr = self.make_move(x, y, player, board)
            if totctr == 0:
                return False
            return True
        elif self.get_board()[y][x] != self.empty_char:
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
    def is_terminal_node(self, player, board=None):
        n = self.board_size
        for y in range(n):
            for x in range(n):
                if board is not None:
                    if self.valid_move(x, y, player, board=board):
                        return False
                else:
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
                    boardTemp2 = self.board_to_numpy(boardTemp)
                    sortedNodes.append((np.array((x, y), dtype=np.int8), self.our_EvalBoard(boardTemp, player)))
        sortedNodes = sorted(sortedNodes, key=lambda node: node[1], reverse=True)
        sortedNodes = np.array([node[0] for node in sortedNodes])
        return sortedNodes

    def get_board(self):
        return self.board