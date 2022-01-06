from q_learning import QLearningModel
from maps import maps


def print_path(path: list):
    """
    Prints the path in a more readable way.

    param path: contains tuples that represents consecutive fields
    type path: list
    """
    to_be_printed = ""
    for row, column in path:
        if to_be_printed != "":
            to_be_printed += f'->({row}, {column})'
        else:
            to_be_printed += f'({row}, {column})'
    print(to_be_printed)


def main(map_number: int = 1):
    """
    Manages the simualtion.

    param map_number: represents the number of the map that should be used in
        simulation
    type map_number: int
    """
    # create the model
    model = QLearningModel(
        maps[map_number]["rows"], maps[map_number]["columns"]
    )
    model.init_map(maps[map_number]["map"])
    # train the model
    model.train()

    # print results
    print(maps[map_number]["map"])
    x0, y0 = model.get_start_location()
    print_path(model.get_shortest_path(x0, y0))


if __name__ == "__main__":
    main()
