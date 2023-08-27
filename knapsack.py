import random
from ga import GA


class Knapsack(GA):
    def __init__(self, items, max_cost, **kwargs):
        self.items = items
        self.max_cost = max_cost
        super().__init__(**kwargs)
    
    def init_population(self):
        return [[random.randint(0,1) for i in range(len(self.items))] for j in range(self.population_size)]
        
    def mutate(self, child):
        for ix, gene in enumerate(child):
            if random.random() < self.mutate_rate:
                child[ix] += random.randint(-1, 1)
        return child
            
    def fitness_score(self, child) -> float:
        score = sum([count*self.items[ix].score for ix, count in enumerate(child)])
        cost = sum([count*self.items[ix].price for ix, count in enumerate(child)])
        if any([gene < 0 for gene in child]) or cost > self.max_cost or score < 0:
            return 0
        return score
        