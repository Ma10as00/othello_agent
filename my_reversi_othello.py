import time
import utils
from players import *
from Board import Board
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(123)
N_RUNS = 25
PLAYER = 1
OPPONENT = 1 if PLAYER == 2 else 2

def BestMove(board_state, player, hmmm=False):
    board = Board()
    board.set_board(board_state)
    max_points = -np.infty
    points = -np.infty
    mx = -1
    my = -1
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
                                         board.minEvalBoard, board.maxEvalBoard, 1, hmmm)
                    if hmmm:
                        print(points)
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
opt = int(input('Select AI Algorithm: '))
if 9 > opt > 0:
    depth = 4
    depthStr = input('Select Search Depth (DEFAULT: 4): ')
    if depthStr != '':
        depth = int(depth)
print('\n1: User 2: AI (Just press Enter for Exit!)')
board = Board()

# file_name = f'8x8_long_train/{board.board_size}x{board.board_size}_model_step_77500.pth'
# file_name = f'8x8_non_sigmoid/{board.board_size}x{board.board_size}_model_step_29999.pth'

intermediates = [x for x in range(0, 15000, 500)] + [14999]

folder_path = '8x8_target_learner_2/'
# folder_path = ''

win_track = np.zeros(len(intermediates))
for i, step in enumerate(intermediates):
    file_name = f'{board.board_size}x{board.board_size}_model_step_{step}.pth'
    # file_name = '8x8_model.pth'
    print(f'\nRunning model {step}\n')
    our_model = utils.load_trained_network(folder_path+file_name, device=device)
    # print(our_model.forward(torch.tensor(utils.get_state(board)).to(device)))
    # exit(0)
    action_board = utils.generate_action_board(board.board_size)

    start = time.time()
    for run in range(N_RUNS):
        board = Board()

        move = 0
        iteration = True
        while iteration:
            for p in range(2):
                # board.print_board()
                player = p + 1
                # print(f'PLAYER: {player} as {board.player_chars[player - 1]}')
                if (board.is_terminal_node(1, copy.deepcopy(board.get_board()))
                        and board.is_terminal_node(2, copy.deepcopy(board.get_board()))):
                    our_score = board.count_board(board.get_board(), player=PLAYER)
                    ai_score = board.count_board(board.get_board(), player=OPPONENT)
                    # print('Player cannot play! Game ended!')
                    print(f'Score User: {our_score}')
                    print(f'Score AI  : {ai_score}\n')
                    iteration = False
                    if our_score - ai_score > 0:
                        win_track[i] += 1
                    break
                if board.is_terminal_node(player, copy.deepcopy(board.get_board())):
                    continue

                if player == PLAYER:  # user's turn
                    # x, y = BestMove(board.get_board(), player)
                    # if not (x == -1 and y == -1):
                    #     temp_board, totctr = board.make_move(x, y, player)
                    #     # print('AI played (X Y): ' + str(x) + ' ' + str(y))
                    #     # print('# of pieces taken: ' + str(totctr))
                    #     board.set_board(temp_board)
                    state = utils.get_state(board)
                    legal_actions = utils.get_legal_action_indices(board, player=PLAYER)

                    if np.sum(legal_actions) == 1:
                        action = np.argmax(legal_actions)
                    else:
                        action_q_values = utils.apply_filter(our_model(torch.tensor(state).to(device)), legal_actions)
                        action = torch.argmax(action_q_values).detach().cpu().numpy()

                        if move < 3:
                            legal_actions[action] = 0
                            a = np.random.choice(np.where(legal_actions == 1)[0])
                            action = np.random.choice([action, a], p=[0.5, 0.5])

                    x_val, y_val = np.where(action_board == action)
                    temp_board, totctr = board.make_move(x_val[0], y_val[0], player=PLAYER)

                    # print('# of pieces taken: ' + str(totctr))
                    board.set_board(temp_board)

                    # while True:
                    #     xy = input('X Y: ')
                    #     if xy == '':
                    #         exit(0)
                    #     (x, y) = xy.split()
                    #     x = int(x)
                    #     y = int(y)
                    #     if board.valid_move(x, y, player):
                    #         temp_board, totctr = board.make_move(x, y, player)
                    #         print('# of pieces taken: ' + str(totctr))
                    #         board.set_board(temp_board)
                    #         break
                    #     else:
                    #         print('Invalid move! Try again!')
                else:  # AI's turn
                    # state = utils.get_state(board)
                    # legal_actions = utils.get_legal_action_indices(board, player=2)
                    #
                    # if np.sum(legal_actions) == 1:
                    #     action = np.argmax(legal_actions)
                    # else:
                    #     action_q_values = utils.apply_filter(our_model(torch.tensor(state).to(device)), legal_actions)
                    #     action = torch.argmax(action_q_values).detach().cpu().numpy()
                    #
                    #     if move < 3:
                    #         legal_actions[action] = 0
                    #         a = np.random.choice(np.where(legal_actions == 1)[0])
                    #         action = np.random.choice([action, a], p=[0.5, 0.5])
                    #         # print(action)
                    #
                    # x_val, y_val = np.where(action_board == action)
                    # temp_board, totctr = board.make_move(x_val[0], y_val[0], player=2)
                    #
                    # # print('# of pieces taken: ' + str(totctr))
                    # board.set_board(temp_board)

                    # state = utils.get_state(board)
                    # # Then get legal actions from current state given the player
                    # legal_actions = utils.get_legal_action_indices(board, player=player)
                    # action = np.random.choice(np.where(legal_actions == 1)[0])
                    #
                    # x_val, y_val = np.where(action_board == action)
                    # temp_board, totctr = board.make_move(x_val[0], y_val[0], player=2)
                    #
                    # # print('# of pieces taken: ' + str(totctr))
                    # board.set_board(temp_board)

                    x, y = BestMove(board.get_board(), player)
                    if not (x == -1 and y == -1):
                        temp_board, totctr = board.make_move(x, y, player)
                        # print('AI played (X Y): ' + str(x) + ' ' + str(y))
                        # print('# of pieces taken: ' + str(totctr))
                        board.set_board(temp_board)
            move += 1

    print(f'Time Elapsed: {round(time.time() - start, 2)}')

# np.savetxt(folder_path + 'winning_track_player_1.txt', win_track)
# win_track = np.loadtxt(folder_path + 'winning_track_player_1.txt')
plt.figure(figsize=(12, 9))
plt.plot(win_track/N_RUNS)
plt.ylim(0, 1)
plt.xticks(range(len(intermediates)), labels=intermediates, rotation=60)
plt.xlabel("Number of Training Iterations")
plt.ylabel("Win Rate")
plt.title(f"Target Function Learner\nWin rate over {N_RUNS} games throughout training")
plt.grid()

plt.show()
