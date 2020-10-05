"""
Tic Tac Toe Player
"""

import math, copy, random

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if sum([i.count(EMPTY) for i in board]) % 2 == 1:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    result = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                result.append((i, j))
    return result


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY or action[0] > 2 or action[1] > 2:
        raise Exception("Invalid move")
    result_board = copy.deepcopy(board)
    result_board[action[0]][action[1]] = player(board)
    return result_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    is_X_won = False
    for j in range(3):
        is_X_won = (
            is_X_won
            or all(X == board[j][i] for i in range(3))
            or all(X == board[i][j] for i in range(3))
        )
    is_X_won = (
        is_X_won
        or all(X == board[i][i] for i in range(3))
        or all(X == board[2 - i][i] for i in range(3))
    )
    if is_X_won:
        return X

    is_O_won = False
    for j in range(3):
        is_O_won = (
            is_O_won
            or all(O == board[j][i] for i in range(3))
            or all(O == board[i][j] for i in range(3))
        )
    is_O_won = (
        is_O_won
        or all(O == board[i][i] for i in range(3))
        or all(O == board[2 - i][i] for i in range(3))
    )

    if is_O_won:
        return O
    return EMPTY


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    who_won = winner(board)
    if (
        who_won == X
        or who_won == O
        or (
            who_won == EMPTY
            and all(board[i][j] != EMPTY for j in range(3) for i in range(3))
        )
    ):
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        result = winner(board)
        return 1 if result == X else -1 if result == O else 0
    return EMPTY


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    cur_player = player(board)
    if cur_player == X:
        value = -math.inf
        for action in actions(board):
            score = min_value(result(board, action), -math.inf, math.inf)
            if score > value:
                value = score
                bestMove = action
    else:
        value = math.inf
        for action in actions(board):
            score = max_value(result(board, action), -math.inf, math.inf)
            if score < value:
                value = score
                bestMove = action

    return bestMove


def max_value(board, alpha, beta):
    if terminal(board):
        return utility(board)
    value = -math.inf
    for action in actions(board):
        value = max(value, min_value(result(board, action), alpha, beta))
        alpha = max(alpha, value)
        if alpha > beta:
            break
    return value


def min_value(board, alpha, beta):
    if terminal(board):
        return utility(board)
    value = math.inf
    for action in actions(board):
        value = min(value, max_value(result(board, action), alpha, beta))
        beta = min(value, beta)
        if alpha > beta:
            break
    return value
