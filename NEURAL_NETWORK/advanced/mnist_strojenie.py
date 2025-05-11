! pip install -q keras-tuner


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt

# Załaduj dane (np. MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Przygotuj dane
x_train = x_train[..., tf.newaxis].astype("float32") / 255.0
x_test = x_test[..., tf.newaxis].astype("float32") / 255.0

# Funkcja budująca model
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    
    # Dodaj 1–3 warstwy konwolucyjne
    for i in range(hp.Int("conv_layers", 1, 3)):
        model.add(layers.Conv2D(
            filters=hp.Choice(f"filters_{i}", [32, 64, 128]),
            kernel_size=hp.Choice(f"kernel_size_{i}", [3, 5]),
            activation="relu"
        ))
        model.add(layers.MaxPooling2D(pool_size=2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int("dense_units", min_value=32, max_value=256, step=32),
        activation="relu"
    ))
    model.add(layers.Dropout(rate=hp.Float("dropout", 0.2, 0.5, step=0.1)))
    model.add(layers.Dense(10, activation="softmax"))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Definicja tunera
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    overwrite=True,
    directory="kt_dir",
    project_name="cnn_mnist"
)

# Callback do wczesnego zatrzymania
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# Wyszukiwanie najlepszych hiperparametrów
tuner.search(x_train, y_train,
             epochs=10,
             validation_split=0.2,
             callbacks=[early_stopping],
             verbose=1)

# Pobierz najlepszy model
best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)

# Trenowanie modelu z najlepszymi hiperparametrami
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Ocena modelu
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")




# Wykres dokładności
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.title('Dokładność trenowania')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)
plt.show()

# Wykres straty
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Strata trenowania')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.grid(True)
plt.show()


# Przegląd top wyników
tuner.results_summary()
