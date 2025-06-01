from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Zakładamy, że masz populację 'pop' z osobnikami, każdy z fitness.values = (f1, f2, f3)
f1_vals = [ind.fitness.values[0] for ind in pop]
f2_vals = [ind.fitness.values[1] for ind in pop]
f3_vals = [ind.fitness.values[2] for ind in pop]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(f1_vals, f2_vals, f3_vals, c=f3_vals, cmap='viridis', s=40, alpha=0.8)

# Etykiety osi
ax.set_xlabel("Rastrigin (do min)")
ax.set_ylabel("Rezonans semantyczny (do min)")
ax.set_zlabel("Entropia (do min)")

# Tytuł i kolor
plt.title("Front Pareto (NSGA-II, 3 cele)")
fig.colorbar(sc, label='Entropia')
plt.tight_layout()
plt.show()
