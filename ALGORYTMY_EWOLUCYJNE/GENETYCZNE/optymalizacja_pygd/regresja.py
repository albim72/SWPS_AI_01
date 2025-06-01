import pygad
import numpy as np

# Dane treningowe
X = np.array([[1, 2], [2, 1], [3, 5], [5, 3]])
y = np.array([10, 11, 25, 28])  # dane wyjściowe

def fitness_func(solution, solution_idx):
    predictions = np.dot(X, solution)
    error = np.mean(np.abs(y - predictions))
    return -error  # minimalizujemy błąd

ga_instance = pygad.GA(num_generations=200,
                       num_parents_mating=5,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_genes=X.shape[1],
                       init_range_low=-10,
                       init_range_high=10,
                       mutation_percent_genes=10)

ga_instance.run()
solution, fitness, _ = ga_instance.best_solution()
print(f"Najlepsze współczynniki: {solution}\nBłąd: {-fitness}")
