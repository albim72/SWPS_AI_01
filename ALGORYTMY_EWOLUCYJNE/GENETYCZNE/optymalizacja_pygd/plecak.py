import pygad
import numpy as np

# Dane: wartości i wagi
values = [10, 20, 30, 40, 15]
weights = [1, 3, 4, 5, 2]
capacity = 8

def fitness_func(solution, solution_idx):
    total_weight = np.dot(solution, weights)
    total_value = np.dot(solution, values)
    if total_weight > capacity:
        return 0  # kara za przekroczenie wagi
    return total_value

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=5,
                       fitness_func=fitness_func,
                       sol_per_pop=20,
                       num_genes=len(values),
                       gene_type=int,
                       gene_space=[0, 1],
                       mutation_percent_genes=20)

ga_instance.run()
solution, fitness, _ = ga_instance.best_solution()
print(f"Wybrane przedmioty: {solution}\nŁączna wartość: {fitness}")
