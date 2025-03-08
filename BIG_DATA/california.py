import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Wczytanie danych
# Pobieramy publiczny zbiór danych California Housing, który zawiera informacje o cenach nieruchomości w Kalifornii.
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # Dodajemy kolumnę z cenami mieszkań (wartość docelowa)

# 2. Podział na zbiór treningowy i testowy
# Dane dzielimy na 80% do treningu i 20% do testów, aby móc ocenić model na nieznanych danych.
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2,
                                                    random_state=42)

# 3. Skalowanie cech
# Normalizujemy dane, aby poprawić efektywność algorytmów uczenia maszynowego.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Inicjalizacja modeli
# Wybieramy dwa modele:
# - Gradient Boosting: silny model boostingowy, dobrze radzący sobie na zbiorach danych o złożonych zależnościach.
# - Random Forest: las losowy, który dobrze generalizuje, ale może wymagać większych zasobów obliczeniowych.
models = {
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=4, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
}

# Parametry:
# - n_estimators: liczba drzew decyzyjnych w modelu (500 dla obu modeli).
# - learning_rate (tylko dla Gradient Boosting): tempo uczenia (0.1).
# - max_depth: maksymalna głębokość drzewa decyzyjnego (4 dla Gradient Boosting, 10 dla Random Forest).
# - random_state: zapewnia powtarzalność wyników.
# - n_jobs (-1 dla Random Forest): używa wszystkich dostępnych rdzeni procesora.

# 5. Trenowanie i ewaluacja
# Dla każdego modelu mierzymy czas trenowania i obliczamy błędy MAE oraz RMSE.
results = {}
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)  # Średni błąd bezwzględny
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Pierwiastek z błędu średniokwadratowego

    results[name] = {
        "Train Time (s)": train_time,
        "MAE": mae,
        "RMSE": rmse
    }

# 6. Wyświetlenie wyników
# Wyniki przedstawiamy w formie tabeli zawierającej czas trenowania oraz wartości błędów.
results_df = pd.DataFrame(results).T
print(results_df)

# Ocena wyników:
# - Im niższe wartości MAE i RMSE, tym lepiej model przewiduje ceny nieruchomości.
# - Gradient Boosting często osiąga lepsze wyniki kosztem dłuższego czasu treningu.
# - Random Forest może być bardziej stabilny, ale mniej dokładny na skomplikowanych danych.
