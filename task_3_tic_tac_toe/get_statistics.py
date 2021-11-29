from run import play_game
from time import time
from mini_max import SIZE


def save_data(data: list, file_path: str = "statistics.csv"):
    """
    Function saves given data into a file

    param data: represents data that are needed to be saved
    type data: list

    param file_path: is a path file
    type file_path: str
    """
    try:
        with open(file_path, "w") as file_handle:
            for line in data:
                file_handle.write(line)
            print("Saving data into given file completed.")
    except IsADirectoryError:
        print("Path is a directory.")
    except FileNotFoundError:
        print("File not found.")
    except Exception:
        print("Undetermined error occurred. Please try again.")


def get_statistics(reps: int = 10):
    """
    Function checks mini max algorithm time performance that is dependent from
    the depth of the search.
    Additionally function collects data about the result of the game.
    Data is saved to the list in a CSV format, but uses semicolons instead of
    commas to be more easily processed by Excel.

    param reps: determines how many time the test must be repeted to collect
        data, time result will be divided by the reps value
    type reps: int
    """
    # initialize start values
    max_depth = SIZE ** 2
    max_player_depth = 0
    statistics = []
    header = "max_player_depth;min_player_depth;avg_time;won;lost;draw;"
    header += "total_st_max_pl;total_st_min_pl;if_ab_cut_max_pl;"
    header += "if_ab_cut_min_pl\n"
    statistics.append(header)

    # iterate over every possible depth vale for max player
    while max_player_depth <= max_depth:
        min_player_depth = 0
        # iterate over every possible depth vale for min player
        while min_player_depth <= max_depth:
            # play game reps time and collect the results
            # iterate over if_ab_cut_max_pl is True as well as False
            if_ab_cut_max_pl = True
            for _ in range(2):
                # iterate over if_ab_cut_min_pl is True as well as False
                if_ab_cut_min_pl = True
                for _ in range(2):
                    start_t = time()
                    won_counter = 0
                    lost_counter = 0
                    draw_counter = 0
                    for _ in range(reps):
                        # st_max_pl - number of states checked by max player
                        # st_min_pl - number of states checked by min player
                        result, st_max_pl, st_min_pl = play_game(
                            True, max_player_depth, min_player_depth,
                            if_ab_cut_max_pl=if_ab_cut_max_pl,
                            if_ab_cut_min_pl=if_ab_cut_min_pl
                        )
                        if result == 1:
                            won_counter += 1
                        elif result == -1:
                            lost_counter += 1
                        else:
                            draw_counter += 1
                    # calculate average time per test
                    finish_t = time()
                    test_avg_time = (finish_t - start_t) / reps
                    # add data to the statistics
                    data_row = f"{max_player_depth};{min_player_depth};"
                    data_row += f"{test_avg_time};{won_counter};"
                    data_row += f"{lost_counter};{draw_counter};"
                    data_row += f"{st_max_pl};{st_min_pl};{if_ab_cut_max_pl};"
                    data_row += f"{if_ab_cut_min_pl}\n"
                    statistics.append(data_row)
                    # change use_ab_cut param value for next play
                    if_ab_cut_min_pl = not if_ab_cut_min_pl
                # change use_ab_cut param value for next play
                if_ab_cut_max_pl = not if_ab_cut_max_pl

            min_player_depth += 1
        max_player_depth += 1

    return statistics


if __name__ == "__main__":
    results = get_statistics(1)
    save_data(results)
