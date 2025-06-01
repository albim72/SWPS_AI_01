import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Funkcja celu – Rastrigin
def rastrigin(ind):
    A = 10
    return A * len(ind) + sum(x**2 - A * np.cos(2 * np.pi * x) for x in ind),

# Parametry
DIMENSIONS = 2
POP_SIZE = 100
GENERATIONS = 50
MU = 20      # rodzice
LAMBDA = 80  # potomstwo
BOUNDS = (-5.12, 5.12)

# Konfiguracja DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, BOUNDS[0], BOUNDS[1])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, DIMENSIONS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", rastrigin)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selBest)

pop = toolbox.population(n=MU)
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# Wykonaj strategię ewolucyjną (mu + lambda)
pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox,
                                         mu=MU, lambda_=LAMBDA,
                                         cxpb=0.5, mutpb=0.3,
                                         ngen=GENERATIONS,
                                         stats=stats,
                                         verbose=True)

# Wizualizacja
gens = logbook.select("gen")
min_vals = logbook.select("min")
avg_vals = logbook.select("avg")
max_vals = logbook.select("max")

plt.figure(figsize=(10, 6))
plt.plot(gens, min_vals, label="Minimum")
plt.plot(gens, avg_vals, label="Średnia", linestyle="--")
plt.plot(gens, max_vals, label="Maksimum", linestyle=":")
plt.xlabel("Generacja")
plt.ylabel("Fitness (Rastrigin)")
plt.title("Strategia ewolucyjna (μ + λ)")
plt.legend()
plt.grid(True)
plt.show()
