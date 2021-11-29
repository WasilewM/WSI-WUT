from time import time
from evolution import evolution
from run import getPopulation
import matplotlib.pyplot as plt


def countAverageTime(populationSize: int):
    """
    Function count the average time of work of the
    evolution algorithm in 3 runs

    param populationSize: represents the population size
    type populationSize: int
    """
    population = getPopulation(populationSize)

    averageTime = 0

    for _ in range(3):
        startTime = time()
        evolution(population, populationSize, 0.5, 1, 0.25, 100, 0.2)
        finishTime = time()
        averageTime += finishTime - startTime

    averageTime /= 3

    return averageTime


def generateTimeCharts():
    """
    Generates time(population) chart
    """
    populationSize = 10
    popValues = []
    timeValues = []

    while populationSize <= 10000:
        popValues.append(populationSize)
        timeValues.append(countAverageTime(populationSize))

        populationSize *= 10

    plt.plot(popValues, timeValues)
    plt.xlabel("Population size")
    plt.ylabel("Time [s]")
    plt.show()


generateTimeCharts()
