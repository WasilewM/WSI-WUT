from evolution import birdFunction, evolution
from random import randint
import numpy as np
import matplotlib.pyplot as plt


def getPopulation(
    size: int,
    rangeWidth: int = 100,
    dimension: int = 2,
    isUnique: bool = True
):
    """
    Function creates a random population of a given size and dimension.

    param size: represents the size of the expected population
    type size: int

    param rangeWidth: determines the range from which values will be drawn -
    should be positive
    type size: int

    param dimension: represents the dimension of the expected entities -
    default value 2
    type dimension: int

    param isUnique: determines wheter the population must be unique or not -
    default value is False
    type isUnique: bool
    """
    population = []

    # used to create uniform population if needed
    seed = randint(-rangeWidth, rangeWidth)

    for _ in range(size):
        entity = []
        while len(entity) < dimension:
            if isUnique:
                entity.append(randint(-rangeWidth, rangeWidth))
            else:
                entity.append(seed)
        population.append(np.array(entity))

    population = np.asarray(population)
    return population


def printResults(bestPerGeneration: np.array):
    """
    Function prints besrPerGeneration array as scatter plot chart.

    param bestPerGeneration: array containing bestPerGeneration
    type bestPerGeneration: np.array
    """
    xValues = []
    yValues = []
    zValues = []

    for entity in bestPerGeneration:
        xValues.append(entity[0])
        yValues.append(entity[1])
        zValues.append(birdFunction(entity))

    plt.scatter(x=xValues, y=yValues, c=zValues)
    plt.colorbar()

    plt.xlabel("x parameter")
    plt.ylabel("y parameter")

    plt.plot(xValues, yValues)
    plt.show()


def main():
    """
    Function main() calls other functions such as
    evolution() to execute evolition algorithm
    printResult() to show results
    """
    population = getPopulation(100)

    bestEntity, bestPerGeneration, bestEntityChanges = evolution(
        population, len(population), 1, 100, 0.1, 1000, 0.2
    )

    print(bestEntity)
    printResults(bestPerGeneration)
    printResults(bestEntityChanges)


if __name__ == "__main__":
    main()
