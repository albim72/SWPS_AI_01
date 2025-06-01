import random

def f(x):
    return -x**2 + 4*x

current_x = random.uniform(0, 4)
step_size = 0.1
max_iterations = 100

for _ in range(max_iterations):
    neighbours = [current_x + step_size, current_x - step_size]
    neighbours = [x for x in neighbours if 0 <= x <= 4]

    next_x = max(neighbours, key=f)

    if f(next_x) <= f(current_x):
        break #osiągnięto lokalne maksimum

    current_x = next_x

print(f"Lokalne maksimum: {current_x}")
print(f"f(x) = {f(current_x)}")
