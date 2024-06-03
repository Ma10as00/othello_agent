import copy
import numpy as np


class Board:
    """
    Othello Board class
    Attributes:
        board_size (int): size of the board
        board (2d list): list of the current state of play
        dirx (list int):
        diry (list int):
        empty_char (str):
        player_chars (list str):
        position_value_matrix (np.array):
        minEvalBoard (int):
        maxEvalBoard (int):
    """
    def __init__(self, board_size=8):
        """
        Initializes all attributes, calls self.init_board() to initialize the board
        Arguments:
            board_size (int): default 8, sets the size of the board
        """
        self.board_size = board_size
        self.board = []
        self.dirx = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.diry = [-1, 0, 1, -1, 1, -1, 0, 1]
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
        # self.minEvalBoard = np.sum(self.position_value_matrix[self.position_value_matrix < 0]) - 1
        # self.maxEvalBoard = np.sum(self.position_value_matrix[self.position_value_matrix > 0]) + 1
        self.minEvalBoard = -1  # min - 1
        self.maxEvalBoard = self.board_size * self.board_size + 4 * self.board_size + 4 + 1  # max + 1
        self.init_board()

    def init_board(self):
        """
        Initializes the board to Othello starting position
        """
        n = self.board_size
        self.board = [[self.empty_char for _ in range(n)] for _ in range(n)]
        if n % 2 == 0:
            z = int((n - 2) / 2)
            self.board[z][z] = self.player_chars[0]
            self.board[z + 1][z] = self.player_chars[1]
            self.board[z][z + 1] = self.player_chars[1]
            self.board[z + 1][z + 1] = self.player_chars[0]

    def board_to_numpy(self, board=None):
        """
        Converts a state of play to an np.array with 0s for empty spaces, 1s for player 1, and 2s for player 2
        Arguments:
            board (2d list): of current state of play, default = None
        Return:
            np.ndarray containing aforementioned information
        """
        if board is None:
            board = self.get_board()

        np_board = np.array(board)
        np_board[np_board == self.player_chars[0]] = 1
        np_board[np_board == self.player_chars[1]] = 2
        np_board[np_board == self.empty_char] = 0

        return np.array(np_board, dtype=np.float32)

    def set_board(self, board_state):
        """
        sets the board to the provided board state
        Arguments:
            board_state (2d list): desired state of the board to be set
        """
        self.board = board_state

    def print_board(self):
        """
        prints the board, should be moved to the __str__ special function...
        """
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
        """
        makes a move in provided position for given player, returns a list of what the board would look like
        Arguments:
            row (int): row
            col (int): col
            player (int): player index
            board (2d list): default none, else list containing a temporary board
        Return:
            tuple:
                temp_board (2d list): the state of the board after the move,
                totctr (int): integer value of number of disks flipped
        """
        totctr = 0  # total number of opponent pieces taken
        n = self.board_size
        if board is not None:
            temp_board = copy.deepcopy(board)
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
        """
        Checks if move at provided location for given player is a legal move
        Arguments:
            row (int): row
            col (int): col
            player (int): player index
            board (2d list): default none, else 2d list of state of the board
        Return:
            bool: True if a valid move, False if not a valid move
        """
        n = self.board_size
        if row < 0 or row > n - 1 or col < 0 or col > n - 1:
            return False
        if board is None:
            board = copy.deepcopy(self.get_board())
        if board[row][col] != self.empty_char:
            return False
        _, totctr = self.make_move(row, col, player, board)
        if totctr == 0:
            return False
        return True

    def our_EvalBoard(self, board_state, player, value_function=None):
        """
        Evaluates the board for the given player with the provided value function
        Arguments:
            board_state (2d list): state of the game board
            player (int): player index
            value_function (2d list of shape board_state): default None, can also be array of position values
        Return:
            total value of the board for the given player
        """
        if value_function is None:
            value_function = np.ones((self.board_size, self.board_size), dtype=int)

        value_of_board = np.sum(value_function[np.array(board_state) == self.player_chars[player - 1]])
        return value_of_board

    def their_eval_board(self, board_state, player):
        """Equivalent to the original EvalBoard(board, player). This function util"""
        value_function = np.ones((self.board_size, self.board_size), dtype=int)
        # Edges
        value_function[0, :-1] = 2
        value_function[:-1, 0] = 2
        value_function[-1, :-1] = 2
        value_function[:-1, -1] = 2
        # Corners
        value_function[(0, -1, 0, -1), (0, -1, -1, 0)] = 4

        return self.our_EvalBoard(board_state, player, value_function=value_function)

    def count_board(self, board_state, player):
        """
        Counts the number of disks a given player has at a specific state of the game
        Arguments:
            board_state (2d list): state of the game board
            player (int): player index
        Return:
            int: total number of disks for the given player
        """
        return self.our_EvalBoard(board_state, player)

    # if no valid move(s) possible then True
    def is_terminal_node(self, player, board_state=None):
        """
        Checks the provided board (or current state of the game) for the existence of legal moves.
        Stops when first legal move is found
        Arguments:
            player (int): player index
            board_state (2d list): state of the game board
        Return:
            bool: True if no legal moves, False if any legal moves exist
        """
        if board_state is None:
            board_state = self.get_board()
        n = self.board_size
        for row in range(n):
            for col in range(n):
                if self.valid_move(row, col, player, board=board_state):
                    return False
        return True

    def get_sorted_nodes(self, player, value_function=None):
        """
        Gets a list of actions in decreasing order of the strength of the position
        according to the provided value function
        Arguments:
            player (int): player index
            value_function (2d list of shape self.get_board()): default None, otherwise the desired value function
        Returns:
            np.array: array of actions in the specified order
        """
        n = self.board_size
        sorted_nodes = []
        for row in range(n):
            for col in range(n):
                if self.valid_move(row, col, player):
                    board_temp, totctr = self.make_move(row, col, player)
                    # boardTemp2 = self.board_to_numpy(boardTemp)
                    sorted_nodes.append((np.array((row, col), dtype=np.int8),
                                        self.their_eval_board(board_temp, player)))
        sorted_nodes = sorted(sorted_nodes, key=lambda node: node[1], reverse=True)
        sorted_nodes = np.array([node[0] for node in sorted_nodes])
        return sorted_nodes

    def get_board(self):
        """
        Gets the current state of play
        Return:
            2d list: current state of the game
        """
        return self.board
