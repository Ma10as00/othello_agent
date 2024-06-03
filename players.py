# Reversi/Othello Board Game using Minimax and Alpha-Beta Pruning
# https://en.wikipedia.org/wiki/Reversi
# https://en.wikipedia.org/wiki/Computer_Othello
# https://en.wikipedia.org/wiki/Minimax
# https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
# https://en.wikipedia.org/wiki/Negamax
# https://en.wikipedia.org/wiki/Principal_variation_search
# FB36 - 20160831
import copy

from Board import Board
import utils
import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
                board_temp, totctr = board.make_move(x, y, player)
                if maximizingPlayer:
                    v = max(v, AlphaBeta(board_temp, player, depth - 1, alpha, beta, False))
                    alpha = max(alpha, v)
                else:
                    v = min(v, AlphaBeta(board_temp, player, depth - 1, alpha, beta, True))
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
    board_temp = board.get_board()
    for moves in sortedNodes:
        board_temp, _ = board.make_move(moves[0], moves[1], player, board_temp)
        if maximizingPlayer:
            v = max(v, AlphaBetaSN(board_temp, player, depth - 1, alpha, beta, False))
            alpha = max(alpha, v)
        else:
            v = min(v, AlphaBetaSN(board_temp, player, depth - 1, alpha, beta, True))
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
    board_temp = board.get_board()
    for moves in sorted_nodes:
        board_temp, _ = board.make_move(moves[0], moves[1], player, board_temp)
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


def NegascoutSN(board_state, player, depth, alpha, beta, color, hmmm=False):
    board = Board()
    board.set_board(board_state)
    if hmmm:
        print('hello', alpha, beta)
        board.print_board()
    if depth == 0 or board.is_terminal_node(player):
        if hmmm:
            print('help me')
        return color * board.their_eval_board(board.get_board(), player)
    sorted_nodes = board.get_sorted_nodes(player)
    first_child = True
    board_temp = board.get_board()
    if hmmm:
        print('first move')
        print(sorted_nodes)
    for moves in sorted_nodes:
        board_temp, _ = board.make_move(moves[0], moves[1], player, board_temp)
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


def get_our_move(file_name, board, player, move):
    action_board = utils.generate_action_board(board.board_size)

    state = utils.get_state(board)
    legal_actions = utils.get_legal_action_indices(board, player=player)

    our_model = utils.load_trained_network(file_name, device=device)


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
    return x_val[0], y_val[0]


def our_self_learned_player(board, player, move):
    return get_our_move(file_name='8x8_self_learning_agent_models/8x8_model_step_14999.pth',
                        board=board, player=player, move=move)


def our_positional_learned_player(board, player, move):
    return get_our_move(file_name='8x8_positional_learning_agent_models/8x8_model_step_29999.pth',
                        board=board, player=player, move=move)
