from q_learning import QLearningModel
from maps import maps
import matplotlib.pyplot as plt


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


def create_graphs(steps: list, rewards: list, episodes=1000):
    """
    Function plots 2 graphs
    1) Steps per episode
    2) Reward per episode

    param steps: represents number of agent's steps in each episode
    type steps: list

    param rewards: represents reward gained by the agent in each episode
    type rewards: list

    param episodes: represents numebr of episodes
    type episodes: int
    """
    x_values = [num for num in range(1000)]

    # plot steps graph
    plt.plot(x_values, steps)
    plt.title('Steps per episode')
    plt.xlabel('Eposiodes')
    plt.ylabel('Steps')
    plt.show()

    # plot rewards graph
    plt.plot(x_values, rewards)
    plt.title('Reward per episode')
    plt.xlabel('Eposiodes')
    plt.ylabel('Reward')
    plt.show()


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
    create_graphs(model.get_steps_stats(), model.get_rewards_stats())


if __name__ == "__main__":
    main()
