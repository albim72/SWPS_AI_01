import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Załaduj dane Indian Pima
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 2. Podział na zbiór treningowy/testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Skalowanie danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Budowa prostej sieci neuronowej
model = Sequential([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.1)

# 5. Wyjaśnienia XAI za pomocą SHAP
explainer = shap.KernelExplainer(model.predict, X_train_scaled[:100])  # Użyj części zbioru treningowego
shap_values = explainer.shap_values(X_test_scaled[:10], nsamples=100)

# 6. Wyświetl wyjaśnienie dla pierwszej próbki
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names=X.columns, matplotlib=True)

# 7. Alternatywnie: wykres słupkowy ważności cech
shap.summary_plot(shap_values[0], X_test_scaled[:10], feature_names=X.columns)


# 8. Oryginalna dokładność
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
baseline_acc = accuracy_score(y_test, y_pred)
print(f"Baseline accuracy: {baseline_acc:.4f}")

# 9. Permutation Feature Importance
importances = []
for i in range(X.shape[1]):
    X_test_permuted = X_test_scaled.copy()
    np.random.shuffle(X_test_permuted[:, i])  # permutacja jednej cechy
    y_pred_perm = (model.predict(X_test_permuted) > 0.5).astype("int32")
    acc = accuracy_score(y_test, y_pred_perm)
    drop = baseline_acc - acc
    importances.append(drop)
    print(f"Feature: {columns[i]} | Drop in accuracy: {drop:.4f}")

# 10. Wykres
plt.figure(figsize=(10, 6))
plt.barh(columns[:-1], importances)
plt.xlabel("Spadek dokładności po permutacji cechy")
plt.title("Permutation Feature Importance (dla sieci neuronowej)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()
