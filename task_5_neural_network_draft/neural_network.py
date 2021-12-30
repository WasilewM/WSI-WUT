from os import error
import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        input_neurons=11,
        hidden_neurons=11,
        output_neurons=11,
        learning_rate=0.005
    ):
        self._input_neurons_num = input_neurons
        self._hidden_neurons_num = hidden_neurons
        self._output_neurons_num = output_neurons
        self._init_params()
        self._alpha = learning_rate
        self.error = []

    def _init_params(self):
        # init params for hidden layer
        self._w1 = np.random.randn(
            self._hidden_neurons_num, self._input_neurons_num
        )
        # self._b1 = np.random.randn(self._hidden_neurons_num, 1)
        # init pramas for output layer
        self._w2 = np.random.randn(
            self._output_neurons_num, self._hidden_neurons_num
        )
        # self._b2 = np.random.randn(self._output_neurons_num, 1)

    def sigmoid(self, x, deriv=False):
        if deriv is True:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        return 1 / (1 + np.exp(-x))

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def forward_propagation(self, x):
        # self._z1 = self._w1.dot(x) + self._b1
        self._z1 = self._w1.dot(x)
        self._a1 = self.sigmoid(self._z1)
        # self._z2 = self._w2.dot(self._a1) + self._b2
        self._z2 = self._w2.dot(self._a1)
        self._a2 = self.softmax(self._z2)

    def get_cost(self, y):
        self._cost = np.zeros((y.size, self._output_neurons_num))
        self._cost[np.arange(y.size), y] = 1
        self._cost = self._cost.T
        # return cost.T

    def get_mean_square_error(self):
        current_epoch_error = np.square(self._dz2).mean()
        # print("error:", current_epoch_error)
        self.error.append(current_epoch_error)

    def back_propagation(self, x, y):
        m = y.size
        self.get_cost(y)
        self._dz2 = self._a2 - self._cost
        # self.get_mean_square_error()
        self._dw2 = 1 / m * self._dz2.dot(self._a1.T)
        # self._db2 = 1 / m * np.sum(self._dz2)
        self._dz1 = self._w2.T.dot(self._dz2) * self.sigmoid(
            self._z1, deriv=True
        )
        self._dw1 = 1 / m * self._dz1.dot(x.T)
        # self._db1 = 1 / m * np.sum(self._dz1)

    def update_weights(self):
        self._w1 -= self._alpha * self._dw1
        # self._b1 -= self._alpha * self._db1
        self._w2 -= self._alpha * self._dw2
        # self._b2 -= self._alpha * self._db2

    def train(self, x, y, epochs=100):
        for i in range(epochs):
            _, samples_num = x.shape
            for idx in range(0, samples_num-1):
                print(idx)
                curr_x = x[:, idx:idx+1]
                curr_y = y[idx]
                self.forward_propagation(curr_x)
                self.back_propagation(curr_x, curr_y)
                self.update_weights()

                if idx == samples_num - 2:
                    # print("Epoch: ", i)
                    # print("Accuracy: ", self.test_performance(x, y))
                    self.get_mean_square_error()

    def get_predictions(self):
        return np.argmax(self._a2, 0)

    def get_accuracy(self, predictions, y):
        print(predictions, y)
        return np.sum(predictions == y) / y.size

    def test_performance(self, x, y):
        _, samples_num = x.shape
        correct_answers = 0
        for idx in range(0, samples_num-1):
            curr_x = x[:, idx:idx+1]
            curr_y = y[idx]
            self.forward_propagation(curr_x)
            answer = self.get_predictions()
            if answer == curr_y:
                correct_answers += 1

        return correct_answers / samples_num