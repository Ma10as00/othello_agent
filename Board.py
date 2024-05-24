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

    def board_to_numpy(self, board=None):
        if board is None:
            board = self.get_board()

        np_board = np.array(board)
        np_board[np_board == self.player_chars[0]] = 1
        np_board[np_board == self.player_chars[1]] = 2
        np_board[np_board == self.empty_char] = 0

        return np.array(np_board, dtype=np.float32)

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

    def make_move(self, row, col, player, board=None):
        totctr = 0  # total number of opponent pieces taken
        n = self.board_size
        if board is not None:
            temp_board = board
        else:
            temp_board = copy.deepcopy(self.get_board())
        temp_board[row][col] = self.player_chars[player - 1]
        for dxx, dyy in zip(self.dirx, self.diry):  # 8 directions
            ctr = 0
            for i in range(n):
                dx = row + dxx * (i + 1)
                dy = col + dyy * (i + 1)
                if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                    ctr = 0
                    break
                elif temp_board[dx][dy] == self.player_chars[player - 1]:
                    break
                elif temp_board[dx][dy] == self.empty_char:
                    ctr = 0
                    break
                else:
                    ctr += 1
            for i in range(ctr):
                dx = row + dxx * (i + 1)
                dy = col + dyy * (i + 1)
                temp_board[dx][dy] = self.player_chars[player - 1]
            totctr += ctr
        return temp_board, totctr

    def valid_move(self, row, col, player, board=None):
        n = self.board_size
        if row < 0 or row > n - 1 or col < 0 or col > n - 1:
            return False
        if board is None:
            board = copy.deepcopy(self.get_board())
        # print(board)
        # if board is not None:
        if board[row][col] != self.empty_char:
            return False
        _, totctr = self.make_move(row, col, player, board)
        if totctr == 0:
            return False
        return True
        # elif self.get_board()[row][col] != self.empty_char:
        #     return False
        # boardTemp, totctr = self.make_move(row, col, player)
        # if totctr == 0:
        #     return False
        # return True

    def our_EvalBoard(self, b, player, value_function=None):
        # tot = 0
        if value_function is None:
            value_function = np.ones((self.board_size, self.board_size), dtype=int)
        # n = self.board_size
        # b = np.array(b)
        tot = np.sum(value_function[np.array(b) == self.player_chars[player - 1]])
        # for row in range(n):
        #     for col in range(n):
        #         if b[row][col] == self.player_chars[player - 1]:
        #             tot += value_function[row][col]
        return tot

    def count_board(self, b, player):
        return self.our_EvalBoard(b, player)#, value_function=np.ones((self.board_size, self.board_size), dtype=int))

    # if no valid move(s) possible then True
    def is_terminal_node(self, player, board=None):
        if board is None:
            board = self.get_board()
        n = self.board_size
        for row in range(n):
            for col in range(n):
                if self.valid_move(row, col, player, board=board):
                    return False
        return True

    def get_sorted_nodes(self, player, value_function=None):
        # if not value_function:
        #     value_function = np.ones((self.board_size, self.board_size), dtype=int)
        n = self.board_size
        sorted_nodes = []
        for row in range(n):
            for col in range(n):
                if self.valid_move(row, col, player):
                    board_temp, totctr = self.make_move(row, col, player)
                    # boardTemp2 = self.board_to_numpy(boardTemp)
                    sorted_nodes.append((np.array((row, col), dtype=np.int8),
                                        self.our_EvalBoard(board_temp, player, value_function=value_function)))
        sorted_nodes = sorted(sorted_nodes, key=lambda node: node[1], reverse=True)
        sorted_nodes = np.array([node[0] for node in sorted_nodes])
        return sorted_nodes

    def get_board(self):
        return self.board
