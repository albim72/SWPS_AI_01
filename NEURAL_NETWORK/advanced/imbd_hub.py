import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model, Layer

# === 1. Ładowanie danych IMDB ===
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

# === 2. Przygotowanie danych (batch, shuffle, prefetch) ===
batch_size = 32

train_data = train_data.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# === 3. Opakowanie KerasLayer, aby działał z TF 2.17 ===
class HubTextEmbedding(Layer):
    def __init__(self, hub_url, trainable=True):
        super().__init__()
        self.embedding_layer = hub.KerasLayer(hub_url, trainable=trainable)

    def call(self, inputs):
        return self.embedding_layer(inputs)

# === 4. Budowa modelu ===
hub_url = "https://tfhub.dev/google/nnlm-en-dim50/2"
input_text = Input(shape=(), dtype=tf.string, name="text_input")
embedding = HubTextEmbedding(hub_url)(input_text)
x = Dense(16, activation='relu')(embedding)
output = Dense(1)(x)

model = Model(inputs=input_text, outputs=output)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# === 5. Trening ===
history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=5)

# === 6. Ewaluacja ===
loss, accuracy = model.evaluate(test_data)
print(f"\nLoss: {loss:.4f} | Accuracy: {accuracy:.4f}")

# === 7. Predykcje na próbkach testowych i wykres ===

# Wybieramy 1 batch z danych testowych
example_batch = next(iter(test_data))  # pobierz pierwszą partię danych
text_batch, label_batch = example_batch
test_examples = text_batch[:5]         # weź 5 przykładów
true_labels = label_batch[:5].numpy()

# Predykcje modelu
raw_preds = model.predict(test_examples)
pred_probs = tf.sigmoid(raw_preds).numpy().flatten()

# Dekodowanie recenzji (z bytes do string)
decoded_reviews = [ex.numpy().decode("utf-8") for ex in test_examples]

# === 8. Wizualizacja ===
plt.figure(figsize=(10, 6))
colors = ['green' if p > 0.5 else 'red' for p in pred_probs]
bars = plt.barh(range(len(decoded_reviews)), pred_probs, color=colors)
plt.yticks(range(len(decoded_reviews)), [" ".join(r.split()[:6]) + "..." for r in decoded_reviews])
plt.xlabel("Pozytywność (0 = negatywna, 1 = pozytywna)")
plt.title("Predykcja sentymentu na recenzjach IMDB")
for i, bar in enumerate(bars):
    plt.text(bar.get_width() + 0.02, bar.get_y() + 0.1, f"{pred_probs[i]:.2f}", fontsize=9)
plt.xlim(0, 1)
plt.tight_layout()
plt.show()
