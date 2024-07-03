import numpy as np

class Particle:
    def __init__(self, position, velocity=None):
        self.position = position
        self.velocity = velocity if velocity is not None else np.zeros_like(position)
        self.best_position = position.copy()
        self.best_value = float('inf')

class ParticleSwarmOptimization:
    def __init__(self, objective_function, n_vars, lower_bound, upper_bound, initial_point=None, constraints=None, swarm_size=50, inertia=0.5, cognitive_rate=1.5, social_rate=1.5):
        self.objective_function = objective_function
        self.n_vars = n_vars
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.constraints = constraints if constraints else []
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive_rate = cognitive_rate
        self.social_rate = social_rate
        self.initial_point = initial_point

    def evaluate(self, x):
        return self.objective_function(x)

    def initialize_swarm(self):
        swarm = [Particle(np.random.uniform(self.lower_bound, self.upper_bound, self.n_vars)) for _ in range(self.swarm_size)]
        return swarm

    def optimize(self, max_iterations=1000):
        swarm = self.initialize_swarm()
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

                # Ensure particles stay within bounds
                particle.position = np.clip(particle.position, self.lower_bound, self.upper_bound)

        return global_best_position, global_best_value

# Objective function and constraints
def objective_function(x):
    x1, x2, x3, x4, x5 = x
    return np.exp(x1 * x2 * x3 * x4 * x5) - 0.5 * (x1**3 + x2**3 + 1)**2

def constraint1(x):
    x1, x2, x3, x4, x5 = x
    return x1**2 + x2**2 + x3**2 + x4**2 + x5**2 - 10

def constraint2(x):
    x1, x2, x3, x4, x5 = x
    return x2 * x3 - 5 * x4 * x5

def constraint3(x):
    x1, x2, x3, x4, x5 = x
    return x1**3 + x2**3 + 1
def create_penalty_function(f,ineq_constraints=[(lambda x:0)],eq_constraints=[(lambda x:0)],rk=1):
    return lambda x:(f(x) +rk*sum([max(0,constraint(x))**2 for constraint in ineq_constraints])+sum([rk*constraint(x)**2 for constraint in eq_constraints]))
# Main execution
if __name__ == "__main__":
    # Setup PSO optimizer with problem parameters
    n_vars = 5
    lower_bound = -5
    upper_bound = 5
    optim_points=[]

    for i in range(0, 20):
        print(f"Penalty function {i}:")
        penalty_function = create_penalty_function(objective_function, ineq_constraints=[constraint1, constraint2], eq_constraints=[constraint3], rk=10^i)
        pso_optimizer = ParticleSwarmOptimization(objective_function, n_vars, lower_bound, upper_bound)

        # Perform optimization
        best_solution, best_fitness = pso_optimizer.optimize()
        print(penalty_function(best_solution))
        optim_points.append(best_solution)

    print(optim_points)
