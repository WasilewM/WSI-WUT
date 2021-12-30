import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork


def plot_error_chart(nn: NeuralNetwork):
    y_values = nn.error
    x_values = [i for i in range(len(y_values))]
    plt.plot(x_values, y_values)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()


def run(
    input_neurons_num=11,
    hidden_neurons_num=11,
    output_neurons_num=11,
    alpha=0.1
):
    # init data
    data = pd.read_csv("winequality-red.csv")
    data = np.array(data)
    rows_num, columns_num = data.shape
    np.random.shuffle(data)

    # split data
    # test data
    data_test = data[0:500].T
    y_test = data_test[columns_num-1]
    x_test = data_test[0:columns_num-1]
    # training data
    data_train = data[500:].T
    y_train_float = data_train[columns_num-1]
    y_train = [
        int(num)
        for num in y_train_float
    ]
    y_train = np.array(y_train)
    x_train = data_train[0:columns_num-1]

    # print(x_train[:, 0:1])

    nn = NeuralNetwork(
        input_neurons_num,
        hidden_neurons_num,
        output_neurons_num,
        alpha
    )
    nn.train(x_train, y_train)
    plot_error_chart(nn)
    print(nn.test_performance(x_test, y_test))


if __name__ == "__main__":
    run()
