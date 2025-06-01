import pygad
import numpy as np

num_workers = 6
num_days = 5

def fitness_func(solution, _):
    schedule = np.round(solution).reshape(num_workers, num_days).astype(int)
    fitness = 0

    for i in range(num_workers):
        working_days = sum(schedule[i])
        if 2 <= (num_days - working_days) <= 3:
            fitness += 5
        if not any(schedule[i, j:j+3].sum() == 3 for j in range(num_days - 2)):
            fitness += 5

    # Pokrycie każdego dnia (wszyscy pracownicy razem)
    daily_coverage = schedule.sum(axis=0)
    fitness += sum(2 <= d <= 4 for d in daily_coverage) * 2  # optimum: 2-4 pracowników/dzień

    return fitness

ga_instance = pygad.GA(num_generations=50,
                       num_parents_mating=5,
                       fitness_func=fitness_func,
                       sol_per_pop=20,
                       num_genes=num_workers * num_days,
                       gene_type=int,
                       gene_space=[0, 1],
                       mutation_percent_genes=10)

ga_instance.run()
solution, fitness, _ = ga_instance.best_solution()
print("Najlepszy harmonogram (0=wolne, 1=praca):\n", np.array(solution).reshape(num_workers, num_days))
