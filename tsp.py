import numpy as np
from typing import List, Union
from ga import GA, Individual
from operators import Crossover, Mutation


class TSP(GA):
    def __init__(self, cities, **kwargs):
        self.cities = cities
        super().__init__(**kwargs)

    def init_population(self) -> List[Individual]:
        population = []
        for j in range(self.population_size):
            rand_permutation = list(np.random.permutation(len(self.cities)))
            population.append(Individual(rand_permutation))
        return population

    def crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        children = Crossover.crossover(parent1,
                                       parent2,
                                       Crossover.pm_crossover,
                                       self._generate_num_children)
        return children

    def mutate(self, individual: Individual) -> Individual:
        return Mutation.random_reverse(individual)

    def fitness_func(self, individual: Individual) -> float:
        total_distance = 0
        for i in range(len(individual) - 1):
            city1 = self.cities[individual.genes[i]]
            city2 = self.cities[individual.genes[i+1]]
            total_distance += TSP.euclidean_distance(city1, city2)

        return 1/total_distance

    @staticmethod
    def euclidean_distance(p, q):
        p = np.array(p)
        q = np.array(q)
        return np.sqrt(np.sum((p-q)**2))

    def _generate_num_children(self) -> int:
        """Function to determine the number of children a couple will produce."""
        return np.random.choice(np.arange(1, 4), p=[0.7, 0.2, 0.1])
