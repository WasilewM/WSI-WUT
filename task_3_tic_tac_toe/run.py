from mini_max import (
    if_player_won,
    mini_max_algorithm,
    INFINITY,
    SIZE
)
import numpy as np


def init_game_board():
    """
    Function creates and returns an empty game board.
    """
    game_board_row = [0] * SIZE
    game_board = []
    for _ in range(SIZE):
        game_board.append(game_board_row)

    return np.asarray(game_board)


def print_game_board(game_board):
    """
    Function prints the game board in a more readable way - with Xs and Os as
    players markers instead of numbers 1s and -1s

    param game_board: represents current game board
    type game_board: np.array
    """
    to_be_printed = ""
    for row in game_board:
        for field in row:
            if field == 1:
                to_be_printed += "O "
            elif field == -1:
                to_be_printed += "X "
            else:
                to_be_printed += "- "
        to_be_printed += "\n"
    print(to_be_printed)


def play_game(
    max_player_starts: bool,
    max_player_depth: int = 3,
    min_player_depth: int = 3,
    max_player_random: bool = False,
    min_player_random: bool = False,
    if_ab_cut_max_pl: bool = True,
    if_ab_cut_min_pl: bool = True
):
    """
    Function manages the gameplay between AI players

    param max_player_starts: answers the question if Max player starts or not
    type max_player_starts: bool

    param max_player_depth: represents the max player depth of search - in
        other words the number of future moves that must be taken into account
        in order to take a deciosion
    type max_player_depth: int

    param min_player_depth: represents the min player depth of search - in
        other words the number of future moves that must be taken into account
        in order to take a deciosion
    type min_player_depth: int

    param max_player_random: answers the question if Max player should take
        random moves - if not, then it plays optimally
    type max_player_random: bool

    param min_player_random: answers the question if Min player should take
        random moves - if not, then it plays optimally
    type min_player_random: bool

    param if_ab_cut_max_pl: answers the question if alpha beta cutting should
        be used for max player
    type if_ab_cut_max_pl: bool

    param if_ab_cut_min_pl: answers the question if alpha beta cutting should
        be used for min player
    type if_ab_cut_min_pl: bool
    """
    # initialize game variables
    round_num = 1
    game_board = init_game_board()
    total_checked_st_max_pl = 0
    total_checked_st_min_pl = 0

    # simulate game rounds
    while round_num <= (SIZE ** 2):
        print(f"Round {round_num}")
        # call mini_max_algorithm to choose move for the player
        best_value, best_row, best_column, checked_st = mini_max_algorithm(
            game_board, max_player_depth, min_player_depth, max_player_starts,
            -INFINITY, INFINITY, max_player_random, min_player_random,
            if_ab_cut_max_pl, if_ab_cut_min_pl
        )

        print("Checked game tree states:", checked_st)

        # choose correct player and apply move returned from mini max algorithm
        if max_player_starts:
            game_board[best_row][best_column] = 1
            total_checked_st_max_pl += checked_st

            print("Max Player, best_value:", best_value)
            print("row:", best_row, "column:", best_column)
            print_game_board(game_board)

            if if_player_won(game_board, max_player_starts):
                print("Max Player has won!")
                return (1, total_checked_st_max_pl, total_checked_st_min_pl)
        else:
            game_board[best_row][best_column] = -1
            total_checked_st_min_pl += checked_st

            print("Min Player, best_value:", best_value)
            print("row:", best_row, "column:", best_column)
            print_game_board(game_board)

            if if_player_won(game_board, max_player_starts):
                print("Min Player has won!")
                return (-1, total_checked_st_max_pl, total_checked_st_min_pl)

        max_player_starts = not max_player_starts
        round_num += 1

    print("Game has finished with a draw")
    print_game_board(game_board)
    return (0, total_checked_st_max_pl, total_checked_st_min_pl)


if __name__ == "__main__":
    play_game(True, 2, 9, if_ab_cut_max_pl=False, if_ab_cut_min_pl=True)
