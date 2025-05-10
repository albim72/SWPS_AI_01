import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tworzymy tensor 3x3x3
tens = np.arange(27).reshape(3, 3, 3)

# Generujemy indeksy dla każdej współrzędnej
x, y, z = np.indices((3, 3, 3))

# Spłaszczamy tensor do listy wartości
values = tens.flatten()

# Normalizujemy wartości do przedziału [0, 1] na potrzeby kolorowania
norm_values = (values - values.min()) / (values.max() - values.min())

# Mapujemy wartości na kolory (używamy cmap 'viridis')
colors = plt.cm.viridis(norm_values)

# Inicjalizujemy figurę 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Tworzymy maskę voxelów – wszystkie widoczne
voxels = np.ones((3, 3, 3), dtype=bool)

# Rysujemy voxele z odpowiadającymi kolorami
ax.voxels(voxels, facecolors=colors.reshape((3, 3, 3, 4)), edgecolor='k')

# Dodajemy etykiety do każdego elementu
for i in range(3):
    for j in range(3):
        for k in range(3):
            ax.text(i + 0.5, j + 0.5, k + 0.5, str(tens[i, j, k]),
                    color='black', ha='center', va='center')

# Opis osi
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Wizualizacja 3D tensora 3x3x3')
plt.tight_layout()
plt.show()
