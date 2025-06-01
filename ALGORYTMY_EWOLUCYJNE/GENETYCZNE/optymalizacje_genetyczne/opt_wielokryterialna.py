from deap import base, creator, tools, algorithms
import numpy as np
import random
import matplotlib.pyplot as plt

def multi_objective(ind):
    rastrigin = 10 * len(ind) + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in ind)
    sum_abs = sum(abs(x) for x in ind)
    return rastrigin, sum_abs

DIMENSIONS = 2
POP_SIZE = 100
GEN = 50
BOUNDS = (-5.12, 5.12)

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # obie funkcje minimalizujemy
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, *BOUNDS)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, DIMENSIONS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", multi_objective)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUNDS[0], up=BOUNDS[1], eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUNDS[0], up=BOUNDS[1], eta=20.0, indpb=1.0/DIMENSIONS)
toolbox.register("select", tools.selNSGA2)

pop = toolbox.population(n=POP_SIZE)

# Inicjalizacja i ocena
for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

for gen in range(GEN):
    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in offspring]

    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= 0.9:
            toolbox.mate(ind1, ind2)
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        del ind1.fitness.values
        del ind2.fitness.values

    for ind in offspring:
        ind.fitness.values = toolbox.evaluate(ind)

    pop = toolbox.select(pop + offspring, k=POP_SIZE)

# WIZUALIZACJA PARETO FRONTU
f1_vals = [ind.fitness.values[0] for ind in pop]
f2_vals = [ind.fitness.values[1] for ind in pop]

plt.figure(figsize=(8, 6))
plt.scatter(f1_vals, f2_vals, c="blue", s=20)
plt.xlabel("Rastrigin (do min)")
plt.ylabel("Suma |x| (do min)")
plt.title("Front Pareto – NSGA-II")
plt.grid(True)
plt.tight_layout()
plt.show()
