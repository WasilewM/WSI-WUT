from q_learning import QLearningModel
from maps import maps
import matplotlib.pyplot as plt


def print_path(path: list, desc: str):
    """
    Prints the path in a more readable way.

    param path: contains tuples that represents consecutive fields
    type path: list

    param desc: short description of used parameters
    type desc: str
    """
    to_be_printed = ""
    required_moves = -1
    for row, column in path:
        if to_be_printed != "":
            to_be_printed += f'->({row}, {column})'
        else:
            to_be_printed += f'({row}, {column})'
        required_moves += 1
    print(to_be_printed, file=open(f'found_paths/path {desc}.txt', 'a'))
    print(f'required_moves: {required_moves}', file=open(f'found_paths/path {desc}.txt', 'a'))


def save_graphs_to_png(steps: list, rewards: list, desc: str, episodes=1000):
    """
    Function creates and saves 2 graphs
    1) Steps per episode
    2) Reward per episode

    param steps: represents number of agent's steps in each episode
    type steps: list

    param rewards: represents reward gained by the agent in each episode
    type rewards: list

    param desc: short description of used parameters to be shown in charts titles
    type desc: str

    param episodes: represents numebr of episodes
    type episodes: int
    """
    x_values = [num for num in range(episodes)]

    fig, axis = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    fig.set_dpi(200)
    fig.suptitle(desc)

    # plot steps graph
    axis[0].scatter(x_values, steps)
    axis[0].set_title('Steps per episode')
    axis[0].set_xlabel('Episodes')
    axis[0].set_ylabel('Steps')
    axis[0].set_ylim([0, 50])

    # plot rewards graph
    axis[1].scatter(x_values, rewards)
    axis[1].set_title('Reward per episode')
    axis[1].set_xlabel('Episodes')
    axis[1].set_ylabel('Reward')
    axis[1].set_ylim([-150, 100])

    plt.savefig(f'graphs_images/{desc}.png')
    plt.close()


def main():
    """
    Function manages the simualtion.
    """

    input_params = (
        # (0.3, 0.3, 0.3, 1000, 3),    # sometimes endless loop - bad params
        # (0.3, 0.5, 0.3, 1000, 3),    # sometimes endless loop - bad params
        # (0.5, 0.5, 0.5, 1000, 3),    # sometimes endless loop - bad params
        (0.65, 0.8, 0.7, 1000, 3),  # weak params, normal map
        (0.9, 0.9, 0.9, 1000, 3),   # optimal params, normal map
        (1., 1., 1., 1000, 3),      # almost always best choice?, normal map
        # (5., 5., 5., 1000, 3),       # very bad params, runtime warning and endless loop

        (0.65, 0.8, 0.7, 1000, 2),  # weak params, difficult map
        (0.9, 0.9, 0.9, 1000, 2),   # optimal params, difficult map
        (1., 1., 1., 1000, 2),      # almost always best choice?, difficult map
    )

    for epsilon, lr, df, e, map_number in input_params:
        desc = f'(epsilon={epsilon} lr={lr} df={df} episodes={e} map={map_number})'
        print(f'Started new simulation {desc}')

        # create the model
        model = QLearningModel(
            maps[map_number]["rows"], maps[map_number]["columns"],
            epsilon=epsilon, lr=lr, df=df
        )
        model.init_map(maps[map_number]["map"])
        # train the model
        model.train(episodes=e)

        # save results
        print(maps[map_number]["map"], file=open(f'found_paths/path {desc}.txt', 'w'))
        x0, y0 = model.get_start_location()
        print_path(model.get_shortest_path(x0, y0), desc)
        # save graphs
        save_graphs_to_png(model.get_steps_stats(), model.get_rewards_stats(), desc, e)


if __name__ == "__main__":
    main()
