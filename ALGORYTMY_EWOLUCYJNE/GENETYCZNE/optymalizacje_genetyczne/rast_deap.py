import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Funkcja Rastrigina
def rastrigin(individual):
    A = 10
    n = len(individual)
    return A * n + sum(x**2 - A * np.cos(2 * np.pi * x) for x in individual),

# Parametry
DIMENSIONS = 2
BOUND_LOW, BOUND_UP = -5.12, 5.12
POP_SIZE = 100
GENERATIONS = 50

# DEAP: definicja środowiska
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, DIMENSIONS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", rastrigin)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Inicjalizacja populacji
pop = toolbox.population(n=POP_SIZE)

# Statystyki i logbook
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()

# Ewolucja
for gen in range(GENERATIONS):
    offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
    fits = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
    pop = toolbox.select(offspring, k=len(pop))

    record = stats.compile(pop)
    logbook.record(gen=gen, **record)

# Wyciągnięcie danych do wykresu
gens = logbook.select("gen")
min_vals = logbook.select("min")
avg_vals = logbook.select("avg")
max_vals = logbook.select("max")

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.plot(gens, min_vals, label="Minimum", linewidth=2)
plt.plot(gens, avg_vals, label="Średnia", linestyle='--')
plt.plot(gens, max_vals, label="Maksimum", linestyle=':')
plt.xlabel("Generacja")
plt.ylabel("Wartość funkcji Rastrigina")
plt.title("Postęp optymalizacji za pomocą algorytmu ewolucyjnego (DEAP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
