import copy
import glob
import os
import time

from matplotlib import pyplot as plt

import utils
from players import *
from Board import Board
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def BestMove(board_state, player, agent_idx, depth=4):
    board = Board()
    board.set_board(board_state)
    max_points = board.minEvalBoard
    points = board.minEvalBoard
    mx = -1
    my = -1
    n = board.board_size
    for row in range(n):
        for col in range(n):
            if board.valid_move(row, col, player):
                board_temp, totctr = board.make_move(row, col, player)
                if agent_idx == 0:
                    points = board.our_EvalBoard(board_temp, player)
                elif agent_idx == 1:
                    points = Minimax(board_temp, player, depth, True)
                elif agent_idx == 2:
                    points = AlphaBeta(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, True)
                elif agent_idx == 3:
                    points = Negamax(board_temp, player, depth, 1)
                elif agent_idx == 4:
                    points = NegamaxAB(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, 1)
                elif agent_idx == 5:
                    points = Negascout(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, 1)
                elif agent_idx == 6:
                    points = AlphaBetaSN(copy.deepcopy(board_temp), player, depth, board.minEvalBoard, board.maxEvalBoard, True)
                elif agent_idx == 7:
                    points = NegamaxABSN(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, 1)
                elif agent_idx == 8:
                    points = NegascoutSN(board_temp, player, depth, board.minEvalBoard, board.maxEvalBoard, 1)
                elif agent_idx == 9:
                    points = board.our_EvalBoard(board_temp, player, board.position_value_matrix)
                if points > max_points:
                    max_points = points
                    mx = row
                    my = col
    return mx, my


print('REVERSI/OTHELLO BOARD GAME')
AI_agent_names = ['EvalBoard', 'MiniMax', 'Minimax w/ Alpha-Beta Pruning', 'Negamax', 'Negamax w/ Alpha-Beta Pruning',
                  'Negascout (Principal Variation Search)', 'Minimax w/ Alpha-Beta Pruning w/ Sorted Nodes',
                  'Negamax w/ Alpha-Beta Pruning w/ Sorted Nodes',
                  'Negascout (Principal Variation Search w/ Sorted Nodes)',
                  'EvalBoard with Position Value Matrix [Eck & van Wezel]']

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
# opt = int(input('Select AI Algorithm: '))
AI_agents = range(10)
# AI_agents = [0, 6, 7, 8, 9]

BOARD_SIZE = 8
TRAIN_STEPS = 2000
iter_vals = [val for val in range(1000, TRAIN_STEPS, 1000)]
# iter_vals.append(TRAIN_STEPS - 1)
action_board = utils.generate_action_board(BOARD_SIZE)

file_names = [f'{BOARD_SIZE}x{BOARD_SIZE}_model_step_{val}.pth' for val in iter_vals]
folder_name = f'{BOARD_SIZE}x{BOARD_SIZE}_models/'

models = []
for file_name in file_names:
    models.append(utils.load_trained_network(folder_name + file_name, device))

scores = np.zeros((len(AI_agents), len(models)))
n_games = 10

for i, model in enumerate(models):
    # if i < 7:
    #     continue
    for j, agent in enumerate(AI_agents):
        # if j != 1:
        #     continue
        score = 0
        # for game_idx in range(n_games):
        board = Board(board_size=BOARD_SIZE)
        print(f'\nPlaying against agent: {AI_agent_names[agent]}, with {iter_vals[i]} training steps')
        running = True
        start = time.time()
        n_flips = 0
        n_steps = 0
        while running:
            for p in range(2):
                # board.print_board()
                player = p + 1
                n_steps += 1
                # print(f'PLAYER: {player} as {board.player_chars[player - 1]}')
                if ((board.is_terminal_node(1, copy.deepcopy(board.get_board()))
                        and board.is_terminal_node(2, copy.deepcopy(board.get_board())))
                        or n_flips == 2 or n_steps == 2 * BOARD_SIZE ** 2):
                    reward = utils.check_reward(board, board.get_board())
                    p1 = board.count_board(board.get_board(), 1)
                    p2 = board.count_board(board.get_board(), 2)
                    print(f'Score p1: {p1}, score p2: {p2}, time elapsed: {round(time.time() - start, 2)}s')
                    # print(reward)
                    # if reward > 0:
                    #     score =
                    # else:
                    #     score = 1
                    score = reward

                    running = False
                    break
                    # print('Player cannot play! Game ended!')
                    # print('Score User: ' + str(board.count_board(board.get_board(), 1)))
                    # print('Score AI  : ' + str(board.count_board(board.get_board(), 2)))
                    # print(f'Time Elapsed: {round(time.time() - start, 2)}')
                    # exit(0)
                if board.is_terminal_node(player, copy.deepcopy(board.get_board())):
                    n_flips += 1
                    continue

                n_flips = 0

                if player == 1:  # user's turn
                    state = utils.get_state(board)
                    legal_actions = utils.get_legal_action_indices(board, player=1)  # , action_board=action_board)

                    action_q_values = utils.apply_filter(model(torch.tensor(state).to(device)), legal_actions)

                    action = torch.argmax(action_q_values).detach().cpu().numpy()

                    x_val, y_val = np.where(action_board == action)
                    temp_board, totctr = board.make_move(x_val[0], y_val[0], player=1)

                    # print('# of pieces taken: ' + str(totctr))
                    board.set_board(temp_board)
                else:  # AI's turn
                    x, y = BestMove(board.get_board(), player, agent_idx=agent)
                    if not (x == -1 and y == -1):
                        temp_board, totctr = board.make_move(x, y, player)
                        # print('AI played (X Y): ' + str(x) + ' ' + str(y))
                        # print('# of pieces taken: ' + str(totctr))
                        board.set_board(temp_board)

        scores[j, i] = score


for i, score_list in enumerate(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(iter_vals, score_list, marker='o', linestyle='-')
    plt.xlabel('Training Iterations')
    plt.ylabel('Margin')
    plt.title(f'Score vs Training Iterations\n{AI_agent_names[i]}')
    plt.grid(True)
    plt.show()
