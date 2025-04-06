import os
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import requests
from io import BytesIO
from PIL import Image

# === 1. Pobieranie danych z Kaggle ===
def download_utkface_from_kaggle():
    if not os.path.exists("UTKFace"):
        print("Pobieranie zbioru UTKFace z Kaggle...")
        os.system("kaggle datasets download -d jangedoo/utkface-new")
        print("Rozpakowywanie...")
        with zipfile.ZipFile("utkface-new.zip", 'r') as zip_ref:
            zip_ref.extractall("UTKFace")
        os.remove("utkface-new.zip")
        print("Dane gotowe w katalogu UTKFace.")
    else:
        print("Dane UTKFace już są dostępne.")

# === 2. Wczytywanie danych ===
def load_utkface_dataset(path, image_size=(64, 64), limit=None):
    images, ages = [], []
    files = os.listdir(path)
    if limit:
        files = files[:limit]
    for filename in files:
        if filename.endswith('.jpg'):
            try:
                age = int(filename.split('_')[0])
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, image_size)
                images.append(img)
                ages.append(age)
            except Exception as e:
                print(f"Błąd przy pliku {filename}: {e}")
    return np.array(images), np.array(ages)

# === 3. Budowa modelu ===
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

# === 4. Predykcja wieku z obrazu URL ===
def predict_age_from_url(image_url, model, image_size=(64, 64)):
    try:
        response = requests.get(image_url)
        img_pil = Image.open(BytesIO(response.content)).convert('RGB')
        img_pil = img_pil.resize(image_size)
        img_np = np.array(img_pil) / 255.0
        img_input = np.expand_dims(img_np, axis=0)
        predicted_age = model.predict(img_input)[0][0]
        plt.imshow(img_np)
        plt.title(f"Przewidywany wiek: {predicted_age:.1f} lat")
        plt.axis('off')
        plt.show()
        return predicted_age
    except Exception as e:
        print(f"Błąd przy przetwarzaniu obrazu: {e}")
        return None

# === 5. GŁÓWNY PIPELINE ===
def main():
    # Pobranie zbioru z Kaggle
    download_utkface_from_kaggle()

    # Wczytanie i przygotowanie danych
    print("Wczytywanie danych...")
    x, y = load_utkface_dataset('UTKFace', image_size=(64, 64), limit=10000)
    x = x / 255.0
    y = y.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Budowa i trening modelu
    print("Trening modelu...")
    model = create_age_regression_model()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    model.fit(x_train, y_train, validation_split=0.1, epochs=10, batch_size=32)

    # Ewaluacja
    loss, mae = model.evaluate(x_test, y_test)
    print(f"\nTest MAE (wiek): {mae:.2f} lat")

    # Predykcja z URL
    test_url = "https://raw.githubusercontent.com/yu4u/age-gender-estimation/master/examples/image1.jpg"
    print("\nPredykcja wieku z obrazu URL:")
    predicted = predict_age_from_url(test_url, model)
    print(f"Model oszacował wiek na: {predicted:.1f} lat")

if __name__ == "__main__":
    main()
