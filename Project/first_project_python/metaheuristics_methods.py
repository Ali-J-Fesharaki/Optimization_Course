import numpy as np

class Particle:
    def __init__(self, position, velocity=None):
        self.position = position
        self.velocity = velocity if velocity is not None else np.zeros_like(position)
        self.best_position = position.copy()
        self.best_value = float('inf')

class ParticleSwarmOptimization:
    def __init__(self, objective_function, n_vars, lower_bound, upper_bound, constraints=None, swarm_size=50, inertia=0.5, cognitive_rate=1.5, social_rate=1.5):
        self.objective_function = objective_function
        self.n_vars = n_vars
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.constraints = constraints if constraints else []
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive_rate = cognitive_rate
        self.social_rate = social_rate

    def evaluate(self, x):
        if self.constraints:
            for constraint in self.constraints:
                if not constraint(x):
                    return 1e6  # Penalty for infeasible solutions
        return self.objective_function(x)

    def optimize(self, max_iterations=100):
        swarm = [Particle(np.random.uniform(self.lower_bound, self.upper_bound, self.n_vars)) for _ in range(self.swarm_size)]
        global_best_position = None
        global_best_value = float('inf')

        for _ in range(max_iterations):
            for particle in swarm:
                current_value = self.evaluate(particle.position)
                if current_value < particle.best_value:
                    particle.best_value = current_value
                    particle.best_position = particle.position.copy()

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particle.position.copy()

                # Update velocity and position
                inertia_term = particle.velocity * self.inertia
                cognitive_term = np.random.rand(self.n_vars) * self.cognitive_rate * (particle.best_position - particle.position)
                social_term = np.random.rand(self.n_vars) * self.social_rate * (global_best_position - particle.position)

                particle.velocity = inertia_term + cognitive_term + social_term
                particle.position += particle.velocity

        return global_best_position, global_best_value


from deap import base, creator, tools, algorithms
import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, objective_function, n_vars, lower_bound, upper_bound, constraints=None, population_size=50, cxpb=0.5, mutpb=0.2):
        self.objective_function = objective_function
        self.n_vars = n_vars
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.constraints = constraints if constraints else []
        self.population_size = population_size
        self.cxpb = cxpb
        self.mutpb = mutpb

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, self.lower_bound, self.upper_bound)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.n_vars)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=self.lower_bound, up=self.upper_bound, eta=0.5, indpb=0.2)
        self.toolbox.register("select", tools.selBest)

    def evaluate(self, individual):
        if self.constraints:
            for constraint in self.constraints:
                if not constraint(individual):
                    return 1e6,  # Penalty for infeasible solutions
        return self.objective_function(individual),

    def optimize(self, ngen=100):
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)

        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

        best_solution = hof[0]
        best_fitness = best_solution.fitness.values[0]

        return best_solution, best_fitness

import numpy as np

class SimulatedAnnealing:
    def __init__(self, objective_function, n_vars, lower_bound, upper_bound, constraints=None, initial_temperature=100.0, cooling_rate=0.95):
        self.objective_function = objective_function
        self.n_vars = n_vars
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.constraints = constraints if constraints else []
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def evaluate(self, x):
        if self.constraints:
            for constraint in self.constraints:
                if not constraint(x):
                    return 1e6  # Penalty for infeasible solutions
        return self.objective_function(x)

    def optimize(self, max_iterations=100):
        current_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.n_vars)
        best_solution = current_solution.copy()
        temperature = self.initial_temperature

        for _ in range(max_iterations):
            proposed_solution = current_solution + np.random.normal(0, 1, self.n_vars)
            current_cost = self.evaluate(current_solution)
            proposed_cost = self.evaluate(proposed_solution)

            if proposed_cost < current_cost or np.random.rand() < np.exp((current_cost - proposed_cost) / temperature):
                current_solution = proposed_solution.copy()

            if self.evaluate(current_solution) < self.evaluate(best_solution):
                best_solution = current_solution.copy()

            temperature *= self.cooling_rate

        return best_solution, self.evaluate(best_solution)


import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_function, n_vars, bounds, constraints=None, population_size=50, crossover_rate=0.7, differential_weight=0.5):
        self.objective_function = objective_function
        self.n_vars = n_vars
        self.bounds = bounds
        self.constraints = constraints if constraints else []
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.differential_weight = differential_weight

    def evaluate(self, x):
        if self.constraints:
            for constraint in self.constraints:
                if not constraint(x):
                    return 1e6  # Penalty for infeasible solutions
        return self.objective_function(x)

    def optimize(self, max_iterations=100):
        population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.population_size, self.n_vars))

        for _ in range(max_iterations):
            for i in range(self.population_size):
                target = population[i]
                indices = np.random.choice(self.population_size, 3, replace=False)
                candidates = population[indices]

                mutant = candidates[0] + self.differential_weight * (candidates[1] - candidates[2])
                mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])

                trial = np.where(np.random.rand(self.n_vars) < self.crossover_rate, mutant, target)
                trial_fitness = self.evaluate(trial)
                target_fitness = self.evaluate(target)

                if trial_fitness < target_fitness:
                    population[i] = trial

        best_solution = population[np.argmin([self.evaluate(x) for x in population])]
        best_fitness = self.evaluate(best_solution)

        return best_solution, best_fitness



from aco import ACO, Graph

class AntColonyOptimization:
    def __init__(self, graph, ant_count=10, generations=100, alpha=1.0, beta=2.0, rho=0.5, q=1.0, strategy='as', heuristic='euclidean'):
        self.graph = graph
        self.ant_count = ant_count
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.strategy = strategy
        self.heuristic = heuristic

    def optimize(self):
        aco = ACO(self.ant_count, self.generations, self.alpha, self.beta, self.rho, self.q, self.strategy, self.heuristic)
        best_path, best_cost = aco.solve(self.graph)

        return best_path, best_cost

from deap import base, creator, tools, algorithms
import random
import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_function, n_vars, lower_bound, upper_bound, constraints=None):
        self.objective_function = objective_function
        self.n_vars = n_vars
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.constraints = constraints if constraints else []

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, self.lower_bound, self.upper_bound)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=self.n_vars)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=self.lower_bound, up=self.upper_bound, eta=0.5, indpb=0.2)
        self.toolbox.register("select", tools.selBest)

    def evaluate(self, individual):
        if self.constraints:
            for constraint in self.constraints:
                if not constraint(individual):
                    return 1e6,  # Penalty for infeasible solutions
        return self.objective_function(individual),

    def optimize(self, population_size=50, cxpb=0.5, mutpb=0.2, ngen=100):
        pop = self.toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)

        pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

        best_solution = hof[0]
        best_fitness = best_solution.fitness.values[0]

        return best_solution, best_fitness

# Example usage of Differential Evolution
if __name__ == "__main__":
    def obj_function(x):
        return np.sum(np.square(x))

    def constraint(x):
        return np.sum(np.square(x)) <= 10  # Example constraint

    de_optimizer = DifferentialEvolution(objective_function=obj_function, n_vars=5, lower_bound=-5.0, upper_bound=5.0, constraints=[constraint])
    best_solution, best_fitness = de_optimizer.optimize(population_size=50, cxpb=0.5, mutpb=0.2, ngen=100)

    print("Differential Evolution:")
    print("Best solution found:", best_solution)
    print("Best fitness:", best_fitness)

