import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# === KONFIGURACJA ===
DATASET_DIR = 'UTKFace'  # ścieżka do folderu ze zdjęciami (np. ./UTKFace/)
IMAGE_SIZE = (64, 64)

# https://www.kaggle.com/datasets/jangedoo/utkface-new
# === ŁADOWANIE DANYCH ===
def load_utkface_dataset(path):
    images = []
    ages = []
    for filename in os.listdir(path):
        try:
            age = int(filename.split('_')[0])
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            ages.append(age)
        except Exception as e:
            print(f"Problem z plikiem {filename}: {e}")
    return np.array(images), np.array(ages)

# === Wczytanie i podział ===
x, y = load_utkface_dataset(DATASET_DIR)
x = x / 255.0  # Normalizacja
y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# === DEFINICJA MODELU ===
def create_age_regression_model(input_shape=(64, 64, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)  # regresja wieku
    ])
    return model

model = create_age_regression_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# === TRENING ===
model.fit(x_train, y_train, validation_split=0.1, epochs=20, batch_size=32)

# === EWALUACJA ===
loss, mae = model.evaluate(x_test, y_test)
print(f"Test MAE (wiek): {mae:.2f} lat")

# === WIZUALIZACJA PREDYKCJI ===
def show_predictions(n=10):
    preds = model.predict(x_test[:n]).flatten()
    actuals = y_test[:n].flatten()
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(x_test[i])
        plt.title(f"Pred: {preds[i]:.0f}\nTrue: {actuals[i]:.0f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_predictions()
