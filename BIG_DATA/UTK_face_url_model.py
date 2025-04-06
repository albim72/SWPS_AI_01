import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import requests
from io import BytesIO
from PIL import Image

# === KONFIGURACJA ===
DATASET_DIR = 'UTKFace'  # Folder ze zbiorami UTKFace (np. ./UTKFace/)
IMAGE_SIZE = (64, 64)

# === ŁADOWANIE DANYCH Z DYSKU ===
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

# === WCZYTYWANIE I PODZIAŁ DANYCH ===
print("Wczytywanie zbioru danych...")
x, y = load_utkface_dataset(DATASET_DIR)
x = x / 255.0
y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# === DEFINICJA MODELU CNN DO REGRESJI WIEKU ===
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
        layers.Dense(1)  # Regresja wieku
    ])
    return model

# === KOMPILACJA I TRENING MODELU ===
print("Budowanie i trenowanie modelu...")
model = create_age_regression_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(x_train, y_train, validation_split=0.1, epochs=20, batch_size=32)

# === EWALUACJA ===
loss, mae = model.evaluate(x_test, y_test)
print(f"Test MAE (wiek): {mae:.2f} lat")

# === FUNKCJA: PREDYKCJA DLA OBRAZU Z URL ===
def predict_age_from_url(image_url, model, image_size=(64, 64)):
    try:
        response = requests.get(image_url)
        img_pil = Image.open(BytesIO(response.content)).convert('RGB')
        img_pil = img_pil.resize(image_size)

        img_np = np.array(img_pil) / 255.0
        img_input = np.expand_dims(img_np, axis=0)

        predicted_age = model.predict(img_input)[0][0]

        # Wizualizacja
        plt.imshow(img_np)
        plt.title(f"Przewidywany wiek: {predicted_age:.1f} lat")
        plt.axis('off')
        plt.show()

        return predicted_age

    except Exception as e:
        print(f"Błąd przy przetwarzaniu obrazu: {e}")
        return None

# === PRZYKŁAD UŻYCIA Z URL ===
image_url = "https://raw.githubusercontent.com/yu4u/age-gender-estimation/master/examples/image1.jpg"
predicted = predict_age_from_url(image_url, model)
print(f"Model oszacował wiek na: {predicted:.1f} lat")
