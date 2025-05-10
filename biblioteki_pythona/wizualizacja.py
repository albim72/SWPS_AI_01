import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Dane: wyniki studentów z 3 przedmiotów
np.random.seed(42)
data = pd.DataFrame({
    'Student': [f'Student {i}' for i in range(1, 101)],
    'Math': np.random.normal(70, 10, 100),
    'Physics': np.random.normal(65, 15, 100),
    'Biology': np.random.normal(75, 12, 100)
})

# 1. Matplotlib – Histogram wyników z matematyki 
plt.figure(figsize=(8,5))
plt.hist(data['Math'], bins=15, color='skyblue', edgecolor='black')
plt.title('Histogram wyników z matematyki')
plt.xlabel('Wynik')
plt.ylabel('Liczba studentów')
plt.grid(True)
plt.show()

# 2. Seaborn – Korelacje między przedmiotami 
plt.figure(figsize=(8,5))
sns.scatterplot(data=data, x='Math', y='Physics', hue='Biology', palette='viridis')
plt.title('Zależność między wynikami z matematyki i fizyki')
plt.show()

# 3. Plotly – Interaktywny wykres 3D 
fig = px.scatter_3d(
    data, x='Math', y='Physics', z='Biology',
    color='Biology', size='Math',
    hover_name='Student',
    title='Interaktywna wizualizacja 3D wyników'
)
fig.show()
