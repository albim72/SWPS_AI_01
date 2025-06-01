import pygad
import numpy as np

# Funkcja celu
def rastrigin_func(solution, solution_idx):
    A = 10
    return A * len(solution) + sum(x**2 - A * np.cos(2 * np.pi * x) for x in solution)

# Parametry GA
ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=10,
                       fitness_func=rastrigin_func,
                       sol_per_pop=20,
                       num_genes=5,
                       init_range_low=-5.12,
                       init_range_high=5.12,
                       mutation_percent_genes=20)

ga_instance.run()
ga_instance.plot_fitness()
solution, fitness, _ = ga_instance.best_solution()
print(f"Najlepsze rozwiązanie: {solution}\nWartość funkcji: {fitness}")
