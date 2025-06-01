import pygad
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Dane - Iris
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Funkcja celu
def fitness_func(solution, solution_idx):
    C = solution[0]
    gamma = solution[1]
    model = SVC(C=C, gamma=gamma)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy  # chcemy maksymalizować

# GA
ga_instance = pygad.GA(num_generations=30,
                       num_parents_mating=5,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_genes=2,
                       init_range_low=0.01,
                       init_range_high=10,
                       mutation_percent_genes=20)

ga_instance.run()
solution, fitness, _ = ga_instance.best_solution()
print(f"Najlepsze hiperparametry: C={solution[0]}, gamma={solution[1]}, Accuracy={fitness}")
