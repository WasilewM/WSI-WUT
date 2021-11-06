from numpy.random import randn
from random import randint, uniform
import numpy as np


def birdFunction(entity: np.array):
    """
    Returns the value of bird function of a given array.

    param entity: represents an entity
    type entity: numpy.array
    """
    x = entity[0]
    y = entity[1]
    return (
        np.sin(x)*(np.exp(1-np.cos(y))**2) +
        np.cos(y)*(np.exp(1-np.sin(x))**2)+(x-y)**2
    )


def markIndividual(entity: np.array):
    """
    Calls function that is being optimized.

    param entity: represents an entity
    type entity: numpy.array
    """
    return birdFunction(entity)


def tournamentSelection(
    population: np.array,
    populationSize: int,
    eliteSize: int,
):
    """
    Function reproduces and selects entities to a new population.

    param population: represents given population - list of numpy.arrays
    type population: numpy.array

    param populationSize: represents given population size
    type populationSize: int

    param eliteSize: tells how many entities from old population must be taken
    directly into the new one
    type eliteSize: int - 0 <= eliteSize <= populationSize
    """

    newPopulation = []
    # elite selection
    sorted_population = sorted(
        population, key=lambda entity: markIndividual(entity), reverse=True
    )
    newPopulation = sorted_population[-eliteSize:]

    # tournament selection - tournament size is 2
    while len(newPopulation) < populationSize:
        # select parents
        candidate_1 = population[randint(0, populationSize - 1)]
        candidate_2 = population[randint(0, populationSize - 1)]

        if markIndividual(candidate_1) < markIndividual(candidate_2):
            newPopulation.append(candidate_1)
        else:
            newPopulation.append(candidate_2)

    newPopulation = np.asarray(newPopulation)
    return newPopulation


def reproduce(
    population: np.array,
    populationSize: int,
    alfa: float,
    crossingProbability: float
):
    """
    Function reproduces population.

    param population: represents given population - list of numpy.arrays
    type population: numpy.array

    param populationSize: represents given population size
    type populationSize: int

    param alfa: represents the ratio of the division of parents: 0 <= alfa <= 1
    type alfa: float

    param crossing probability: determines the probability under which
    crossing occurs
    type crossingProbability: float, should be in range (0, 1)
    """
    # reproduce
    crossedPopulation = []
    while len(crossedPopulation) < populationSize:
        # select parents
        parent_1 = population[randint(0, populationSize - 1)]
        parent_2 = population[randint(0, populationSize - 1)]

        crossing_chance = uniform(0, 1)

        if crossing_chance <= crossingProbability:
            crossedPopulation.append(alfa * parent_1 + (1 - alfa) * parent_2)
            crossedPopulation.append((1 - alfa) * parent_1 + alfa * parent_2)
        else:
            crossedPopulation.append(parent_1)
            crossedPopulation.append(parent_2)

    while len(crossedPopulation) > populationSize:
        crossedPopulation.pop()

    crossedPopulation = np.asarray(crossedPopulation)
    return crossedPopulation


def mutatePopulation(population: np.array, sigma: float):
    """
    Function is responsible for introducing mutations into population.

    param population: represents given population - list of numpy.arrays
    type population: numpy.array

    param sigma: represents mutation power
    type sigma: float
    """
    newPopulation = []

    for entity in population:
        new_entity = []
        for attribute in entity:
            # draw if and in which direction mutatuion occurs
            mutation = randn()
            mutation *= sigma
            attribute = attribute + mutation
            new_entity.append(attribute)

        newPopulation.append(new_entity)

    newPopulation = np.asarray(newPopulation)
    return newPopulation


def findBestEntity(bestMark: float, population: np.array):
    """
    Function finds the entity with the best mark in the given population.

    param bestMark: mark of the best entity
    type bestMark: float

    param population: represents given population - list of numpy.arrays
    type population: numpy.array
    """

    for entity in population:
        if markIndividual(entity) == bestMark:
            return entity


def evolution(
        population: np.array,
        populationSize: int,
        sigma: float,
        eliteSize: int,
        alfa: float,
        iterations: int,
        crossingProbability: float
):
    """
    Function manages the evolution - evolution algorithm.

    param population: represents given population - list of numpy.arrays
    type population: numpy.array

    param populationSize: represents given population size
    type populationSize: int

    param sigma: represents mutation power
    type sigma: float

    param eliteSize: tells how many entities from old population must be taken
    directly into the new one
    type eliteSize: int - 0 <= eliteSize <= populationSize

    param alfa: represents the ratio of the division of parents: 0 <= alfa <= 1
    type alfa: float

    param iterations: represens the number of generations to be checked
    type iterations: int - iterations >= 0

    param crossing probability: determines the probability under which
    crossing occurs
    type crossingProbability: float, should be in range (0, 1)
    """
    # setting initial value to iterator
    iterNum = 0

    # saving the best mark
    bestMark = min((
        markIndividual(p)
        for p in population
    ))
    # saving the best entity
    bestEntity = findBestEntity(bestMark, population)

    # creating a list that will represent the change of the best
    # entity in the population
    bestPerGeneration = []
    bestPerGeneration.append(bestEntity)
    bestEntityChanges = []
    bestEntityChanges.append(bestEntity)

    while iterNum < iterations:
        # tournament selection
        population = tournamentSelection(
            population,
            populationSize,
            eliteSize
        )
        # crossing
        population = reproduce(
            population,
            populationSize,
            alfa,
            crossingProbability
        )
        # mutate
        population = mutatePopulation(population, sigma)
        # mark the population and find the best enity in the population
        currPopulationBest = min((
            markIndividual(p)
            for p in population
        ))
        # save the best
        if (currPopulationBest < bestMark):
            bestMark = currPopulationBest
            bestEntity = findBestEntity(bestMark, population)
            bestEntityChanges.append(bestEntity)

        bestPerGeneration.append(
            findBestEntity(currPopulationBest, population)
        )
        iterNum += 1

    bestPerGeneration = np.asarray(bestPerGeneration)
    return bestEntity, bestPerGeneration, bestEntityChanges
