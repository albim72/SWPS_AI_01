
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load labeled CSV
df = pd.read_csv("opinions_labeled_1000.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Tokenization
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=20, padding='post')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(np.array(X_train), np.array(y_train), epochs=10, validation_data=(np.array(X_test), np.array(y_test)))

# Evaluation
pred_probs = model.predict(np.array(X_test))
pred_labels = [1 if p > 0.5 else 0 for p in pred_probs]
print(classification_report(y_test, pred_labels))

# Save model and tokenizer config
model.save("sentiment_model.h5")

# ---------------------------
# Prediction on new samples
# ---------------------------
test_sentences = [
    "I absolutely love this product!",
    "Terrible experience, I will never buy this again.",
    "Great quality and fast delivery.",
    "Completely useless and a waste of money.",
    "Highly recommend it to everyone!",
    "Awful packaging and very slow shipping.",
    "This is the best purchase I made this year!",
    "Not satisfied, the item broke after two days.",
    "Super helpful and kind customer service!",
    "Disappointed with how it turned out.",
    "Amazing quality, exceeded my expectations!",
    "The product stopped working after a week.",
    "Really fast shipping and nice packaging.",
    "Customer service was not helpful at all.",
    "It's okay, nothing special.",
    "Absolutely fantastic, I'm impressed!",
    "Won’t be buying from this seller again.",
    "Love it! Will order again soon.",
    "Worst purchase I've ever made.",
    "Top-notch quality and very reliable."
]

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=20, padding='post')

preds = model.predict(test_padded)

print("\nPredictions on 20 new sentences:")
for i, sent in enumerate(test_sentences):
    label = "POSITIVE" if preds[i][0] > 0.5 else "NEGATIVE"
    print(f"{label} ({preds[i][0]:.2f}): {sent}")
