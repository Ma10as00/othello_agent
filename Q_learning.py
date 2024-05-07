import copy

import numpy as np
from Board import Board


class OthelloBoard:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)  # 0: empty, 1: player 1, 2: player 2
        self.board[3:5, 3:5] = [[2, 1], [1, 2]]  # Initial Othello configuration
        self.current_player = 1  # Player 1 starts

    def print_board(self):
        for row in self.board:
            print(row)

    def get_legal_moves(self):
        legal_moves = []
        for i in range(8):
            for j in range(8):
                if self.is_legal_move(i, j):
                    legal_moves.append((i, j))
        return legal_moves

    def is_legal_move(self, i, j):
        if self.board[i, j] != 0:
            return False
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                if self.check_direction(i, j, di, dj):
                    return True
        return False

    def check_direction(self, i, j, di, dj):
        opponent = 3 - self.current_player
        x, y = i + di, j + dj
        while 0 <= x < 8 and 0 <= y < 8 and self.board[x, y] == opponent:
            x, y = x + di, y + dj
        if 0 <= x < 8 and 0 <= y < 8 and self.board[x, y] == self.current_player:
            return True
        return False

    def make_move(self, i, j):
        if not self.is_legal_move(i, j):
            raise ValueError("Invalid move")
        self.board[i, j] = self.current_player
        self.flip_pieces(i, j)
        self.current_player = 3 - self.current_player  # Switch player

    def flip_pieces(self, i, j):
        opponent = 3 - self.current_player
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                if self.check_direction(i, j, di, dj):
                    self.flip_direction(i, j, di, dj)

    def flip_direction(self, i, j, di, dj):
        opponent = 3 - self.current_player
        x, y = i + di, j + dj
        while 0 <= x < 8 and 0 <= y < 8 and self.board[x, y] == opponent:
            self.board[x, y] = self.current_player
            x, y = x + di, y + dj

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {}  # Q-value table
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        # state_key = tuple(map(tuple, state))  # Convert NumPy array to tuple of tuples
        # return self.q_table.get((state_key, action), 0.0)
        # try:
        # print()
        if self.q_table.get((state.copy().tobytes(), action.copy().tobytes())) is None:
            self.q_table[(state.copy().tobytes(), action.copy().tobytes())] = 0.0

        return self.q_table.get((state.copy().tobytes(), action.copy().tobytes()))
        # except TypeError:
        #     return self.q_table.get((state.tobytes(), action.tobytes()))

    def update_q_value(self, state, action, reward, next_state):
        othello = Board()
        state = othello.board_to_numpy(state.copy())
        # print(state)
        np_next_state = othello.board_to_numpy(next_state.copy())
        state_key = tuple(map(tuple, state))  # Convert NumPy array to tuple of tuples
        next_state_key = tuple(map(tuple, next_state))
        current_q = self.get_q_value(state, action)
        othello.set_board(next_state)
        max_future_q = max([self.get_q_value(np_next_state, a) for a in self.get_legal_actions(othello)])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[(state.copy().tobytes(), action.copy().tobytes())] = new_q

    @staticmethod
    def get_legal_actions(board):
        return board.get_sorted_nodes(1)

    def choose_action(self, board, state):
        if np.random.uniform(0, 1) < self.epsilon:
            actions = self.get_legal_actions(board)
            # print(actions.shape)
            # my_range = np.arange(0, actions.shape[0], 1)
            # print(my_range)
            # choice = np.random.choice(np.arange(0, actions.shape[0], 1))
            # print(choice)
            return 0, actions[np.random.choice(np.arange(0, actions.shape[0], 1))]
        else:
            actions = self.get_legal_actions(board)
            print(actions)
            q_values = {}
            for a in actions:
                bited = a.copy().tobytes()
                q_values[(state.copy().tobytes(), bited)] = self.get_q_value(state, a.copy())
            print(actions)
            print(q_values)
            # q_values = {(state.copy().tobytes(), a.copy().tobytes()): self.get_q_value(state, a.copy()) for a in actions}
            return max(q_values, key=q_values.get)

    def train(self, episodes=1000):
        # othello = OthelloBoard()
        othello = Board()
        othello.init_board()

        for _ in range(episodes):
            state = copy.deepcopy(othello.get_board())
            np_state = othello.board_to_numpy(state).copy()
            action = np.frombuffer(self.choose_action(othello, np_state)[1], dtype=np.int8)
            print('hi')
            print(action)
            othello.make_move(action[0], action[1], player=1)
            next_state = copy.deepcopy(othello.get_board())
            # Reward function (you can define your own)
            reward = self.calculate_reward(state, action, next_state)
            self.update_q_value(state, action, reward, next_state)

    @staticmethod
    def calculate_reward(state, action, next_state):
        # You can define your own reward function based on game outcomes
        othello = Board()
        if othello.is_terminal_node(1, next_state):
            n_Os = np.sum(next_state == othello.player_chars[1])
            n_Xs = np.sum(next_state == othello.player_chars[0])
            if n_Os > n_Xs:
                return 1
            elif n_Os == n_Xs:
                return 0.5
        return 0  # Placeholder for simplicity


my_q_learning_agent = QLearningAgent()
print(my_q_learning_agent.q_table)
np.random.seed(42)
my_q_learning_agent.train(100)

print(my_q_learning_agent.q_table)