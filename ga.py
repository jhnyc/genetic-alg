from typing import List
import random
from collections import deque
import numpy as np
from abc import ABC, abstractmethod


class GA(ABC):
    def __init__(self, population_size, num_generation, mutate_rate, survival_rate, shuffle_mating=True):
        self.population_size = population_size
        self.num_generation = num_generation
        self.mutate_rate = mutate_rate
        self.survival_rate = survival_rate
        self.shuffle_mating = shuffle_mating
        
        self.population_fitness_scores = []
        self.evolution_history = []
    
    @abstractmethod
    def init_population(self):
        pass    
    
    def run(self):
        population = self.init_population()
        for gen in range(self.num_generation):
            # Find the fittest survivors from the current population
            parents = self.fittest_parents(population)
            # Survivors produce their next generation along with genetic mutations
            children = self.produce_next_generation(parents)
            # New population will be the children & any of their parents who are superior
            population = self.combine_children_and_better_parents(parents, children)
            self.log_evolution_history()
            
        return population
            
    
    def combine_children_and_better_parents(self, parents, children):
        """Find parents who are better than their offsprings, and combine with their children to form the new population.
        This ensures the best person is always in the population.
        """
        children_fitness_scores = [self.fitness_score(c) for c in children]
        best_child_fitness_score = max(children_fitness_scores)
        
        better_parents_ix = filter(lambda ix: self.population_fitness_scores[ix] > best_child_fitness_score, range(len(parents)))
        new_population = children + [parents[ix] for ix in better_parents_ix] 
        self.population_fitness_scores = children_fitness_scores + [self.population_fitness_scores[ix] for ix in better_parents_ix]
        return new_population
        
        
    
    def fittest_parents(self, population):
        """Reduce the population to only the fittest members, who will be the parents that breed the next generation."""
        if not self.population_fitness_scores:
            self.population_fitness_scores = [self.fitness_score(p) for p in population]
        
        survivor_ix = sorted(range(len(population)), key=lambda ix: self.population_fitness_scores[ix], reverse=True)
        num_parents = int(self.survival_rate * len(population))
        self.population_fitness_scores = [
            self.population_fitness_scores[ix] for ix in survivor_ix][:num_parents]
        return [population[i] for i in survivor_ix][:num_parents]
    
    
    def generate_num_children(self):
        """Function to determine the number of children a couple will produce."""
        return np.random.choice(np.arange(1, 4), p=[0.7, 0.2, 0.1])
    
    
    def mate(self, parent1, parent2) -> list:
        """A pair of parents produce their children."""
        children = []
        num_children = self.generate_num_children()
        
        for _ in range(num_children):
            pivot = random.randint(0, len(parent1)-1)
            if random.random() > 0.5:
                children += self.born_child(parent1[:pivot] + parent2[pivot:])
            else:
                children += self.born_child(parent2[:pivot] + parent1[pivot:])
                
        return children
    
    def born_child(self, genes):
        """Determine whether a mutated child is eligible."""
        mutated_genes = self.mutate(genes)
        if self.fitness_score(mutated_genes):
            return [mutated_genes]
        return []
        
    def produce_next_generation(self, parents):
        """All parents in the population produce the next generation children."""
        next_generation = []
        
        while len(next_generation) < self.population_size:
            mating_queue = deque(list(range(len(parents))))
            if self.shuffle_mating:
                random.shuffle(mating_queue)
            
            while len(mating_queue) > 1:
                parent1_ix, parent2_ix = mating_queue.popleft(), mating_queue.popleft()
                children = self.mate(parents[parent1_ix], parents[parent2_ix])
                next_generation += children
            
        return next_generation
    
    
    def log_evolution_history(self):
        """Log evolution history in terms of the best fitness score of each generation."""
        best_score = max(self.population_fitness_scores)
        self.evolution_history.append(best_score)
    
    @abstractmethod
    def mutate(self, child):
        """Apply mutation to a child."""
        pass
    

    @abstractmethod
    def fitness_score(self, person) -> float:
        """Compute fitness score of a person."""
        pass
    