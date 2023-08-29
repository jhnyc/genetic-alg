import random
import numpy as np
from typing import List, Union
from ga import GA, Individual
from operators import Crossover, Mutation


class Knapsack(GA):
    def __init__(self, items, max_cost: Union[int, float], **kwargs):
        self.items = items
        self.max_cost = max_cost
        super().__init__(**kwargs)
        

    def init_population(self) -> List[Individual]:
        population = []
        for j in range(self.population_size):
            individual = Individual([random.randint(0, 1) for i in range(len(self.items))])
            population.append(individual)
        return population
    
    def crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        children = Crossover.random_crossover(parent1, parent2, self._generate_num_children)
        return children

    def mutate(self, individual: Individual) -> Individual:
        mutated_individual = Mutation.discrete_mutate(individual, 
                                                      mutation_range=(-1, 1), 
                                                      mutate_rate=self.mutate_rate)
        return mutated_individual

    def fitness_func(self, individual: Individual) -> float:
        score = sum([count*self.items[ix].score for ix,
                    count in enumerate(individual)])
        cost = sum([count*self.items[ix].price for ix,
                   count in enumerate(individual)])
        if any([gene < 0 for gene in individual.genes]) or cost > self.max_cost or score < 0:
            return 0
        return score

    def _generate_num_children(self) -> int:
        """Function to determine the number of children a couple will produce."""
        return np.random.choice(np.arange(1, 4), p=[0.7, 0.2, 0.1])
