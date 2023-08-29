from typing import List, Callable, Tuple, Union
import random
import numpy as np
from ga import Individual


class Mutation:
    @staticmethod    
    def base_mutate(individual: Individual, mutate_func: Callable, mutate_rate: float) -> Individual:
        for ix, gene in enumerate(individual.genes):
            if random.random() < mutate_rate:
                individual.genes[ix] += mutate_func()
        return individual
    
    @staticmethod    
    def discrete_mutate(individual: Individual, mutation_range: Tuple[int,int], mutate_rate: float) -> Individual:
        mutate_func = lambda: np.random.randint(*mutation_range)
        return Mutation.base_mutate(individual, mutate_func, mutate_rate)
    
    @staticmethod    
    def continuous_mutate(individual: Individual, mutation_range: Tuple, mutate_rate: float) -> Individual:
        mutate_func = lambda: np.random.uniform(*mutation_range)
        return Mutation.base_mutate(individual, mutate_func, mutate_rate)
    
    def random_reverse(individual: Individual) -> Individual:
        # Select segment to be reversed
        cx_point1, cx_point2 = random.sample(range(len(individual)), k=2)
        if cx_point1 > cx_point2:
            cx_point1, cx_point2 = cx_point2, cx_point1
        
        # Reverse the selected segment
        individual.genes[cx_point1:cx_point2+1] = individual.genes[cx_point1:cx_point2+1][::-1]
        return individual

class Crossover:
    @staticmethod
    def crossover(parent1: Individual, parent2: Individual, crossover_func:Callable, gen_num_children_func: Callable=None) -> List[Individual]:
        num_children = gen_num_children_func() if gen_num_children_func else 1
        children = []
        for _ in range(num_children):
            children.append(crossover_func(parent1, parent2))
        return children
    
    @staticmethod
    def random_crossover(parent1: Individual, parent2: Individual) -> Individual:
        """1 point crossover at a random pivot point."""
        pivot = random.randint(0, len(parent1)-1)
        if random.random() > 0.5:
            return Individual(parent1.genes[:pivot] + parent2.genes[pivot:])
        return Individual(parent2.genes[:pivot] + parent1.genes[pivot:])
    
    @staticmethod
    def mid_crossover(parent1: Individual, parent2: Individual) -> Individual:
        """1 point crossover at a the middle point."""
        pivot = len(parent1)//2
        if random.random() > 0.5:
            return Individual(parent1.genes[:pivot] + parent2.genes[pivot:])
        return Individual(parent2.genes[:pivot] + parent1.genes[pivot:])
    
    def uniform_crossover(parent1: Individual, parent2: Individual):
        """Each bit is chosen from either parent with equal probability."""
        combined_genes = [random.choice([i, j]) for i, j in zip(parent1.genes, parent2.genes)]
        return Individual(combined_genes)
        
    
    def pm_crossover(parent1: Individual, parent2: Individual) -> Individual:
        """Algorithm walkthrough: https://www.youtube.com/watch?v=EZg-l2FF-JM&ab_channel=DEEBAKANNAN"""
        child_genes = [None] * len(parent1)
        
        cx_point1, cx_point2 = random.sample(range(len(parent1)), k=2)
        if cx_point1 > cx_point2:
            cx_point1, cx_point2 = cx_point2, cx_point1
        
        # Copy the segment from parent1 to child genes
        child_genes[cx_point1:cx_point2+1] = parent1.genes[cx_point1:cx_point2+1]
        
        # Fill in remaining genes such that each gene is unique
        for i in range(len(parent2)):
            if child_genes[i] is not None:
                continue
            
            gene = parent2.genes[i]
            is_p2 = True # Alternate between parent1 and parent2
            while gene in child_genes:
                ix = child_genes.index(gene)
                gene = parent1.genes[ix] if is_p2 else parent2.genes[ix]
                is_p2 = not is_p2
                
            child_genes[i] = gene
               
        return Individual(child_genes)