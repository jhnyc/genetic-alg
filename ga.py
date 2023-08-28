from typing import List, Callable, Union
import random
from collections import deque
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Individual:
    def __init__(self, genes: List[Union[int,float]]):
        self.genes = genes
        
    def __hash__(self):
        return hash(str(self.genes))
        
    def __eq__(self, other):
        return str(self.genes) == str(other.genes)
    
    def __iter__(self):
        return iter(self.genes)
    
    def __len__(self):
        return len(self.genes)
    
    def __str__(self):
        return f"Individual({self.genes})"
    
    def __repr__(self):
        return f"Individual({self.genes})"
    

class GA(ABC):
    def __init__(self, population_size: int, 
                 num_generation: int, 
                 selection_func: Callable, 
                 mutate_rate: float, 
                 survival_rate: float, 
                 shuffle_mating: bool = True, 
                 **selection_kwargs):
        self.population_size = population_size
        self.num_generation = num_generation
        self.mutate_rate = mutate_rate
        self.survival_rate = survival_rate
        self.shuffle_mating = shuffle_mating
        self.selection_func = self.get_selection_func(selection_func, **selection_kwargs)
        self.evolution_history = []
        self.fitness_cache = {}
        self.best_individual = None
    
    @abstractmethod
    def init_population(self) -> List[Individual]:
        pass    
    
    def get_selection_func(self, selection_func: str, **selection_kwargs) -> Callable:
        ga_selection = GASelection(self.get_fitness_score, self.survival_rate, **selection_kwargs)
        selection_method_mapping = {
            "survival": ga_selection.survival_selection,
            "tournament": ga_selection.tournament_selection,
            "rank": ga_selection.rank_based_selection,
            "roulette": ga_selection.roulette_wheel_selection
        }
        selection_method = selection_method_mapping.get(selection_func)
        if selection_method is None:
            raise ValueError("Invalid selection function specified.")
        
        return selection_method
        
    
    def run(self):
        population = self.init_population()
        for gen in range(self.num_generation):
            # Find the fittest survivors from the current population
            parents = self.selection_func(population)
            # Survivors produce their next generation along with genetic mutations
            children = self.produce_next_generation(parents)
            # New population will be the children & any of their parents who are superior
            population = self.combine_children_and_better_parents(parents, children)
            self.log_evolution_history(population)
            
        return population
            
    
    def combine_children_and_better_parents(self, parents:List[Individual], children:List[Individual]):
        """Find parents who are better than their offsprings, and combine with their children to form the new population.
        This ensures the best person is always in the population.
        """
        children_fitness_scores = [self.get_fitness_score(c) for c in children]
        best_child_fitness_score = max(children_fitness_scores)
        
        better_parents = list(filter(lambda i: self.get_fitness_score(i) > best_child_fitness_score, parents))
        new_population = children + better_parents
        return new_population
        
    
    @abstractmethod
    def crossover(self, parent1:Individual, parent2:Individual) -> List[Individual]:
        """A pair of parents produce their children."""
        pass
    
    def born_child(self, individual: Individual):
        """Determine whether a mutated child is eligible to be born."""
        mutated_individual = self.mutate(individual)
        if self.get_fitness_score(mutated_individual):
            return [mutated_individual]
        return []
        
    def produce_next_generation(self, parents: List[Individual]):
        """All parents in the population produce the next generation children."""
        next_generation = []
        
        while len(next_generation) < self.population_size:
            mating_queue = deque(list(range(len(parents))))
            if self.shuffle_mating:
                random.shuffle(mating_queue)
            
            while len(mating_queue) > 1:
                parent1_ix, parent2_ix = mating_queue.popleft(), mating_queue.popleft()
                children = self.crossover(parents[parent1_ix], parents[parent2_ix])
                next_generation += children
            
        return next_generation
    
    
    def log_evolution_history(self, population: List[Individual]):
        """Log evolution history in terms of the best fitness score of each generation."""
        best_individual, best_score = max([(i, self.get_fitness_score(i)) for i in population], key=lambda x: x[1])
        self.best_individual = best_individual
        self.evolution_history.append(best_score)
    
    @abstractmethod
    def mutate(self, individual:Individual) -> Individual:
        """Apply mutation to a child."""
        pass
    
    def get_fitness_score(self, individual:Individual) -> float:
        """A wrapper of fitness function with caching."""
        if individual in self.fitness_cache:
            return self.fitness_cache[individual]
        score = self.fitness_func(individual)
        self.fitness_cache[individual] = score
        return score

    @abstractmethod
    def fitness_func(self, individual: Individual) -> float:
        """Compute fitness score of an individual. The higher the better."""
        pass
    
    def plot_evolution_history(self):
        plt.plot(self.evolution_history)
        plt.title("Evolution History")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.show()


class GASelection:
    def __init__(self, fitness_func: Callable, 
                 survival_rate: float, 
                 tournament_size: int = 3):
        self.fitness_func = fitness_func
        self.survival_rate = survival_rate
        self.tournament_size = tournament_size
        
        
    def survival_selection(self, population: List[Individual]):
        """Select the fittest individuals to survive and become parents for the next generation."""
        population_sorted = sorted(population, key=lambda i: self.fitness_func(i), reverse=True)
        num_parents = int(self.survival_rate * len(population))
        selected_parents = population_sorted[:num_parents]
        return selected_parents
    
    def tournament_selection(self, population: List[Individual]):
        """Perform tournament selection to choose parents for the next generation."""
        num_parents = int(self.survival_rate * len(population))
        selected_parents = []

        for _ in range(num_parents):
            tournament_size = min(self.tournament_size, len(population))
            contestants = random.sample(population, tournament_size)
            winner = max(contestants, key=lambda i: self.fitness_func(i))
            selected_parents.append(winner)

        return selected_parents
    
    def rank_based_selection(self, population: List[Individual]):
        """Select parents based on rank-based probabilities."""
        num_parents = int(self.survival_rate * len(population))
        population_sorted = sorted(population, key=lambda i: self.fitness_func(i), reverse=True)
        
        num_population = len(population)
        rank_probabilities = [i / (num_population * (num_population + 1) / 2) for i in range(1, num_population + 1)]
        
        selected_parents = np.random.choice(population_sorted, p=rank_probabilities, size=num_parents, replace=False)
        
        return selected_parents
        
    def roulette_wheel_selection(self, population: List[Individual]):
        """Select parents based on probabilities calculated from fitness scores."""
        num_parents = int(self.survival_rate * len(population))
        
        fitness_scores = [self.fitness_func(i) for i in population]
        total_fitness_score = sum(fitness_scores)
        probabilities = [s/total_fitness_score for s in fitness_scores]
        
        selected_parents = np.random.choice(
            population, p=probabilities, size=num_parents, replace=False)
        return selected_parents
    