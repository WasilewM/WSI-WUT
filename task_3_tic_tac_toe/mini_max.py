import numpy as np
from random import randint
INFINITY = int(1e9)
SIZE = 3
UKNOWN = 0
# heuristics values from lecture
# the value of the field represents the number of winning combinations
# in which the field is included
FIELD_IMPORTNACE = [
        [3, 2, 3],
        [2, 4, 2],
        [3, 2, 3]
    ]
FIELD_IMPORTNACE = np.asarray(FIELD_IMPORTNACE)


def if_player_won(node: np.array, is_max_player: bool):
    """
    Function answers the question if given player has won.

    param node: represents state of the game - node in a game tree
    type node: np.array

    param is_max_player: answers the question if given player is max player
        if not, then given player is min player
    type is_max_player: bool
    """
    # 1 represents max player sign (O or X)
    # -1 represents min player sign (X or O)
    sign = 1 if is_max_player else -1

    # check rows - check if one of rows is a winning combination
    for row in node:
        player_won = True
        for field in row:
            if field != sign:
                player_won = False
                break

        if player_won is True:
            # the entire row belongs to the player; therefore, it wins
            return True

    # check columns - check if one of columns is a winning combination
    for column_num in range(SIZE):
        player_won = True
        for row_num in range(SIZE):
            if node[row_num][column_num] != sign:
                player_won = False
                break

        if player_won is True:
            # the entire column belongs to the player; therefore, it wins
            return True

    # check diagonals check if one of columns is a winning combination
    # check the first diagonal - upper left to lower right corners
    player_won = True
    for row_num in range(SIZE):
        column_num = row_num
        if node[row_num][column_num] != sign:
            player_won = False
            break

    if player_won is True:
        # the diagonal belongs to the player; therefore, it wins
        return True

    # check the second diagonal - upper right to lower left corners
    player_won = True
    for value in range(SIZE):
        idx = SIZE - value - 1
        if node[idx][idx] != sign:
            player_won = False
            break

    if player_won is True:
        # the diagonal belongs to the player; therefore, it wins
        return True

    # no winning combination found; therefore, player has not win not lost yet
    return False


def is_game_finished(node: np.array):
    """
    Function answers the question if game is already finished or not

    param node: represents state of the game - node in a game tree
    type node: np.array
    """
    if if_player_won(node, True):
        # Max player has won
        return 100
    elif if_player_won(node, False):
        # Min player has won
        return -100
    else:
        # noone wins yet
        return 0


def get_board_status(node: np.array):
    """
    Function calculates the state of the game board.

    param node: represents state of the game - node in a game tree
    type node: np.array
    """
    # checking if game already is won or lost
    is_finished = is_game_finished(node)
    if is_finished in (-100, 100):
        return is_finished

    # game is still in progres; therefore, calculate board status
    result = 0
    # iteration over rows of the game board
    row_num = 0
    for row in node:
        # iteration over columns in chosen row
        column_num = 0
        for field in row:
            # include every field of the game board into the result of the
            # board field value determines whether this field is occupied
            # by Max or Min player or if it is empty
            # field value is multiplied by its importance and then added
            # to the result
            result += field * FIELD_IMPORTNACE[row_num][column_num]
            column_num += 1
        row_num += 1

    return result


def get_heuristics(node, is_max_player):
    """
    Function calculates and returns the best available action at the moment.

    param node: represents state of the game - node in a game tree
    type node: np.array

    param is_max_player: answers the question if given player is max player
        if not, then given player is min player
    type is_max_player: bool
    """
    # define default value for best guess if nothing can be found
    # it is returned from the function only if the board is full,
    # otherwise better value will be found and returned
    best_guess = (-INFINITY, 0, 0, 0) if is_max_player else (INFINITY, 0, 0, 0)

    # chech game board for possible moves and choose best
    # iteration over rows of the game board
    row_num = 0
    for row in node:
        # iteration over columns in chosen row
        column_num = 0
        for field in row:
            # check if field is empty
            if field == 0:
                # field is empty and can be used
                changed_node = np.copy(node)
                changed_node[row_num][column_num] = 1 if is_max_player else -1
                # get mark for node
                node_mark = get_board_status(changed_node)
                # check if it is better then current best
                # for Max player if it is Max player's turn
                if (is_max_player) and (node_mark > best_guess[0]):
                    best_guess = (node_mark, row_num, column_num, 1)
                # for Min player if it is Min player's turn
                elif (not is_max_player) and (node_mark < best_guess[0]):
                    best_guess = (node_mark, row_num, column_num, 1)
            column_num += 1
        row_num += 1

    return best_guess


def is_board_full(node):
    """
    Function answers the question if board is full

    param node: represents state of the game - node in a game tree
    type node: np.array
    """
    # iteration over rows of the game board
    for row in node:
        # iteration over columns in chosen row
        for field in row:
            if field == 0:
                return False
    return True


def get_available_fields(node: np.array):
    """
    Function returns available fields

    param node: represents state of the game - node in a game tree
    type node: np.array
    """
    available_fields = []
    row_num = 0
    # iterate over each field and check if it is available
    # if it is available add it to the list
    for row in node:
        column_num = 0
        for field in row:
            if field == 0:
                available_fields.append((row_num, column_num))
            column_num += 1
        row_num += 1

    return np.asarray(available_fields)


def get_random_move(node: np.array, checked_states):
    """
    Function returns random move

    param node: represents state of the game - node in a game tree
    type node: np.array

    param checked_states: represents the number of checked states
    type checked_states: int
    """
    # get available moves
    available_moves = get_available_fields(node)
    idx = randint(0, len(available_moves) - 1)
    # return chosen move
    x, y = available_moves[idx]
    return (UKNOWN, x, y, checked_states)


def mini_max_algorithm(
    node: np.array,
    max_player_depth: int,
    min_player_depth: int,
    is_max_player: bool,
    alpha: int,
    beta: int,
    max_player_random: bool = False,
    min_player_random: bool = False,
    if_ab_cut_max_pl: bool = True,
    if_ab_cut_min_pl: bool = True
):
    """
    Function is an impementation of the mini max algorithm.
    Additionally it uses alpha beta cutting technique / algorithm in order to
    shorten time needed to take deciosion.

    param node: represents state of the game - node in a game tree
    type node: np.array

    param max_player_depth: represents the max player depth of search - in
        other words the number of future moves that must be taken into account
        in order to take a deciosion
    type max_player_depth: int

    param min_player_depth: represents the min player depth of search - in
        other words the number of future moves that must be taken into account
        in order to take a deciosion
    type min_player_depth: int

    param max_player_random: answers the question if max player should take a
        random deciosion or not
    param max_player_random: bool

    param min_player_random: answers the question if min player should take a
        random deciosion or not
    param min_player_random: bool

    param if_ab_cut_max_pl: answers the question if alpha beta cutting should
        be used for max player
    type if_ab_cut_max_pl: bool

    param if_ab_cut_min_pl: answers the question if alpha beta cutting should
        be used for min player
    type if_ab_cut_min_pl: bool
    """
    # initialize checked_states counter
    checked_states = 1

    # check if player has won
    if if_player_won(node, is_max_player):
        if is_max_player:
            return (100, -1, -1, checked_states)
        else:
            return (-100, -1, -1, checked_states)

    # check if player lost or if it is a draw
    if (is_game_finished(node) != 0) or (is_board_full(node)):
        return (get_board_status(node), -1, -1, checked_states)

    # check if Max player must make a random move
    if is_max_player and max_player_random:
        return get_random_move(node, checked_states)

    # check if Min player must make a random move
    if (not is_max_player) and min_player_random:
        return get_random_move(node, checked_states)

    if is_max_player and (max_player_depth == 0):
        return get_heuristics(node, is_max_player)

    if (not is_max_player) and (min_player_depth == 0):
        return get_heuristics(node, is_max_player)

    # game is still in progress
    best_row_num = 0
    best_column_num = 0

    if is_max_player:
        best_value = -INFINITY
        row_num = 0
        # iteration over rows of the game board
        for row in node:
            column_num = 0
            # iteration over columns in chosen row
            for field in row:
                if field == 0:
                    # field is available; therefore, create child node
                    # and evaluate it
                    child_node = np.copy(node)
                    # 1 represents Max Player's field
                    child_node[row_num][column_num] = 1
                    child_value, _, _, child_st_num = mini_max_algorithm(
                        child_node, max_player_depth-1, min_player_depth,
                        False, alpha, beta, max_player_random,
                        min_player_random, if_ab_cut_max_pl, if_ab_cut_min_pl
                    )
                    # increase the number of checked game tree states
                    checked_states += child_st_num
                    # determine if child node is better then current best
                    if best_value < child_value:
                        best_value = child_value
                        best_row_num = row_num
                        best_column_num = column_num

                    # alpha - beta algorithm - cutting subtrees
                    if if_ab_cut_max_pl:
                        alpha = max(alpha, best_value)
                        if beta <= alpha:
                            break
                # iterating over columns
                column_num += 1
            # iterating over rows
            row_num += 1

        return (best_value, best_row_num, best_column_num, checked_states)
    else:
        best_value = INFINITY
        row_num = 0
        # iteration over rows of the game board
        for row in node:
            column_num = 0
            # iteration over columns in chosen row
            for field in row:
                if field == 0:
                    # field is available; therefore, create child node
                    # and evaluate it
                    child_node = np.copy(node)
                    # -1 represents Min Player's field
                    child_node[row_num][column_num] = -1
                    child_value, _, _, child_st_num = mini_max_algorithm(
                        child_node, max_player_depth, min_player_depth-1,
                        True, alpha, beta, max_player_random,
                        min_player_random, if_ab_cut_max_pl, if_ab_cut_min_pl
                    )
                    # increase the number of checked game tree states
                    checked_states += child_st_num
                    # determine if child node is better then current best
                    if best_value > child_value:
                        best_value = child_value
                        best_row_num = row_num
                        best_column_num = column_num

                    # alpha - beta algorithm - cutting subtrees
                    if if_ab_cut_min_pl:
                        beta = min(beta, best_value)
                        if beta <= alpha:
                            break
                # iterating over columns
                column_num += 1
            # iterating over rows
            row_num += 1

        return (best_value, best_row_num, best_column_num, checked_states)
