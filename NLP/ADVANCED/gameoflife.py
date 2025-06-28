import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def update(frame, img, grid):
    new_grid = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                         grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N]))
            if grid[i, j] == 1:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = 0
            else:
                if total == 3:
                    new_grid[i, j] = 1
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img,

N = 60  # Rozmiar planszy
grid = np.random.choice([0, 1], N*N, p=[0.75, 0.25]).reshape(N, N)

fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='binary')
ax.axis('off')

ani = FuncAnimation(fig, update, fargs=(img, grid), frames=100, interval=120, blit=True)
plt.show()
