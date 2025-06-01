import pygad
import numpy as np

# Uproszczona plansza 4x4, zera = puste miejsca
initial_board = np.array([
    [1, 0, 0, 0],
    [0, 0, 3, 4],
    [0, 0, 0, 0],
    [4, 0, 0, 2]
])

def fitness_func(solution, _):
    board = initial_board.copy()
    flat_indices = np.where(board == 0)
    board[flat_indices] = np.round(solution).astype(int)
    
    score = 0
    for i in range(4):
        score += len(set(board[i, :]))  # unikalność w wierszu
        score += len(set(board[:, i]))  # unikalność w kolumnie
    return score

num_variables = np.count_nonzero(initial_board == 0)
ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=5,
                       fitness_func=fitness_func,
                       sol_per_pop=20,
                       num_genes=num_variables,
                       init_range_low=1,
                       init_range_high=4,
                       mutation_percent_genes=20)

ga_instance.run()
solution, fitness, _ = ga_instance.best_solution()
print("Fitness końcowy:", fitness)
