import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametry
N = 30   # Liczba boidów
L = 100  # Rozmiar planszy

# Inicjalizacja pozycji i wektorów prędkości
pos = np.random.rand(N, 2) * L
vel = (np.random.rand(N, 2) - 0.5) * 10

def limit_speed(v, max_speed=4):
    speed = np.linalg.norm(v)
    if speed > max_speed:
        v = v / speed * max_speed
    return v

def update(frame):
    global pos, vel
    new_vel = np.zeros_like(vel)
    for i in range(N):
        # Reguły: align, cohere, separate
        diff = pos - pos[i]
        dist = np.linalg.norm(diff, axis=1)
        mask = (dist > 0) & (dist < 20)
        # Separation
        sep = -np.sum(diff[mask] / (dist[mask][:,None] + 1e-5), axis=0)
        # Alignment
        align = np.mean(vel[mask], axis=0) if np.any(mask) else np.zeros(2)
        # Cohesion
        coh = np.mean(diff[mask], axis=0) if np.any(mask) else np.zeros(2)
        v = vel[i] + 0.03*sep + 0.05*align + 0.01*coh
        new_vel[i] = limit_speed(v)
    vel = new_vel
    pos[:] = (pos + vel) % L
    scat.set_offsets(pos)

fig, ax = plt.subplots(figsize=(6,6))
scat = ax.scatter(pos[:,0], pos[:,1], s=50)
ax.axis('off')
ax.set_xlim(0, L)
ax.set_ylim(0, L)

ani = FuncAnimation(fig, update, frames=200, interval=70)
plt.show()
