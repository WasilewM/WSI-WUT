import numpy as np


class BayesianGenerator:
    def __init__(self, probabilities: dict):
        """
        Constructor for class BayesianGenerator.

        param probabilities: represents probabilities of variables/nodes in
            bayesian network
        type probabilities: dict
        """
        self._probabilities = probabilities

    def get_probabilites(self):
        """
        Getter for probabilities attribute.
        """
        return self._probabilities

    def generate_data(self, samples: int = 100000):
        """
        Function generates data samples consistent with the bayesian network
        distribution.

        param samples: number of samples to be generated
        type samples: int
        """
        data = np.zeros((samples, 1))   # init data with its initial shape
        variables = []  # list for varaiables / nodes name
        for var in self.get_probabilites():     # iterate over variables
            variables.append(var)
            dependencies, probabilities = self.get_probabilites()[var]
            if dependencies == []:
                # variables / node is not dependent from other variables
                column = self.generate_data_column(probabilities[0], samples)
                data = np.column_stack((data, column))
            else:
                # variables / node is dependent from some of other variables
                depts_data = np.zeros((samples, 1))   # dependencies
                for dep in dependencies:
                    column_index = variables.index(dep)
                    # because the first column is column of zeros
                    column_index += 1
                    depts_data = np.column_stack((
                        depts_data, data[:, column_index]
                    ))
                # delete column of zeros
                depts_data = np.delete(depts_data, 0, axis=True)
                # get values for current variable based on already generated
                # values
                column = self.generate_dependent_column(
                    probabilities, depts_data, samples
                )
                # append data
                data = np.column_stack((data, column))
        # delete column of zeros
        data = np.delete(data, 0, axis=True)
        return variables, data

    def generate_data_column(self, true_prob: int, samples: int):
        """
        Function generates values for current, independent, variable.
        Generation is consistent with the bayesian network distribution.

        param true_prob: probability of tru for current variable - should be
            between 0 and 1 exclusive
        type true_prob: int

        param samples: number of samples
        type samples: int
        """
        return np.random.choice(
            (0, 1),
            (samples, 1),
            p=(1 - true_prob, true_prob)
        )

    def generate_dependent_column(self, probabilities, depts_data, samples):
        """
        Function generates values for current, dependent, variable.
        Generation is consistent with the bayesian network distribution.
        """
        column = np.zeros((samples, 1))     # initialize column shape
        row_num = 0
        for row in depts_data:
            prob_idx = 0
            factor = 1
            # Iterate over already generated variables and based on those
            # variables choose probability. Probabilities are organised:
            # FF, TF, FT, TT
            # FFF, TFF, FTF, TTF, FFT, TFT, FTT, TTT
            # etc ...
            for var in row:
                prob_idx += var * factor
                factor *= 2
            prob = probabilities[int(prob_idx)]
            # based on calculated probability index choose value
            column[row_num] = np.random.choice((0, 1), p=(1-prob, prob))
            row_num += 1
        return column
