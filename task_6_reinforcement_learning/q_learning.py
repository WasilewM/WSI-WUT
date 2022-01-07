import numpy as np


class QLearningModel:
    def __init__(self, rows_num: int, columns_num: int):
        """
        Constructor for class QLearningModel.

        param rows_num: represents the number of rows of the agent's
            environment
        type rows_num: int

        param columns_num: represents the number of columns of the agent's
            environment
        type columns_num: int
        """
        # add 2 additional rows and columns to create boarders around the
        # environment to prevent the agent from going outside the environment
        self._rows_num = rows_num + 2
        self._colums_num = columns_num + 2

        # set up rewards / penalties
        self._obstacle_reward = -100
        self._goal_reward = 100
        self._path_reward = -1

        # set up values for learinin process
        self._epsilon = 0.9     # probability of choosing best action
        self._lr = 0.9          # learning rate
        self._df = 0.9          # discount factor

        # set up agent's actions - environment is a 2D board using urban
        # metrics
        self._actions = {
            0: 'UP',
            1: 'RIGHT',
            2: 'DOWN',
            3: 'LEFT'
        }

        # initialize Q-values
        self._q_values = np.zeros(
            (self._rows_num, self._colums_num, len(self._actions))
        )

        # initialize rewards to obstacle reward - later fields with path and
        # destination will be updated to their appropriate rewards
        self._rewards = np.full(
            (self._rows_num, self._colums_num), self._obstacle_reward
        )

        # initialize lists to collect data about learing / training process
        # after each episode
        self._step_stats = []
        self._rewards_stats = []

    def init_map(
        self,
        environment: str,
        start: str = 'S',
        finish: str = 'F',
        path: str = '.'
    ):
        """
        Update rewards for path and destination / goal fields.

        param environment: represents the map of the environment
        type environment: str

        param start: represents the sign of the start field
        type start: str

        param finish: represents the sign of the finish / goal field
        type finish: str

        param path: represents the sign of legal fields - fields on which the
            agent can legally move
        param path: str
        """
        idx = 0     # iterator for the environment
        for i in range(1, self._rows_num - 1):  # iterate over rows
            for j in range(1, self._colums_num - 1):    # iterate over columns
                if environment[idx] == start:
                    self._rewards[i, j] = self._path_reward
                    # save start location for leter use - especially for
                    # training
                    self._start_location = (i, j)
                elif environment[idx] == path:
                    self._rewards[i, j] = self._path_reward
                elif environment[idx] == finish:
                    self._rewards[i, j] = self._goal_reward
                # move iterator to next field
                idx += 1
            # skip new line symbol of the environment
            idx += 1

    def get_start_location(self):
        """
        Getter for start_location attribute.
        """
        return self._start_location

    def get_next_action(self, curr_row: int, curr_column: int, is_test=False):
        """
        Returns next move of the agent.
        Has 2 modes determined by is_test param. If is_test equals True than
        the best possible action should be returned. Otherwise next action is
        determined by the probability.

        param curr_row: represents current row of the agent
        type curr_row: int

        param curr_column: represents current column of the agent
        type curr_column: int

        param test: determines whether it is a test mode or not
        type test: bool
        """
        # if random number is lower than epsilon then agent should perform the
        # most profitable action
        if np.random.random() < self._epsilon or is_test:
            return int(np.argmax(self._q_values[curr_row, curr_column]))
        else:
            # otherwise return random action
            return int(np.random.randint(len(self._actions)))

    def get_next_location(self, curr_row: int, curr_column: int, action: int):
        """
        Returns next location of the agent.

        param curr_row: represents current row of the agent
        type curr_row: int

        param curr_column: represents current column of the agent
        type curr_column: int

        param action: represents index of action
        type action: int
        """
        if self._actions[action] == 'UP':
            return curr_row - 1, curr_column
        elif self._actions[action] == 'RIGHT':
            return curr_row, curr_column + 1
        elif self._actions[action] == 'DOWN':
            return curr_row + 1, curr_column
        elif self._actions[action] == 'LEFT':
            return curr_row, curr_column - 1
        else:
            return curr_row, curr_column

    def is_terminal_state(self, row_num, column_num):
        """
        Answers the question if current state is a terminal one or not.
        Terminal state is goal / finish or wall of the environments. State is
        determined bu row and column numbers.

        param row_num: represents current row of the agent
        type row_num: int

        param column_num: represents current column of the agent
        type column_num: int
        """
        if self._rewards[row_num, column_num] == -1:
            return False
        else:
            return True

    def train(self, episodes=1000):
        """
        Function manages the process of training.

        param eposiodes: represents the number of episodes
        type eposiodes: int
        """
        # iterate over given number of eposiodes
        for _ in range(episodes):
            curr_row, curr_column = self.get_start_location()

            epsiode_total_steps = 0
            epsiode_total_reward = 0

            # move around the environment until run into terminal state
            while not self.is_terminal_state(curr_row, curr_column):
                # get action
                action = self.get_next_action(curr_row, curr_column)

                # get new position
                new_row, new_column = self.get_next_location(
                    curr_row, curr_column, action
                )

                # get reward
                reward = self._rewards[new_row, new_column]
                epsiode_total_reward += reward

                # update Q-values
                old_q_value = self._q_values[curr_row, curr_column, action]
                q_max = np.max(self._q_values[new_row, new_column])
                temporal_diff = reward + self._df * q_max - old_q_value

                self._q_values[curr_row, curr_column, action] = (
                    old_q_value + self._lr * temporal_diff
                )

                # move to new location
                curr_row = new_row
                curr_column = new_column

                # increment steps value
                epsiode_total_steps += 1

            # collect data about episode
            self._step_stats.append(epsiode_total_steps)
            self._rewards_stats.append(epsiode_total_reward)

    def get_shortest_path(self, row_num, column_num):
        """
        Return the shortes path between given field and goal determined by the
        environment. Given field is determined by row and column numbers.

        param row_num: represents the row of the given field
        type row_num: int

        param column_num: represents the column of the given field
        type column_num: int
        """
        path = []
        path.append((row_num, column_num))  # save data

        # move around the environment until run into terminal state
        while not self.is_terminal_state(row_num, column_num):
            # get action
            action = self.get_next_action(row_num, column_num, is_test=True)
            # get new position
            row_num, column_num = self.get_next_location(
                row_num, column_num, action
            )
            path.append((row_num, column_num))  # save data

        return path

    def get_steps_stats(self):
        """
        Getter for step_stats attribute.
        """
        return self._step_stats

    def get_rewards_stats(self):
        """
        Getter for rewards_stats attribute.
        """
        return self._rewards_stats
