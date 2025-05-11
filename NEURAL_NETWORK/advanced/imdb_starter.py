import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model, Layer

# Opakowujemy hub.KerasLayer w Layer, aby działał z KerasTensor
class HubTextEmbedding(Layer):
    def __init__(self, hub_url, trainable=True):
        super().__init__()
        self.embedding_layer = hub.KerasLayer(hub_url, trainable=trainable)

    def call(self, inputs):
        return self.embedding_layer(inputs)

# 1. Input
input_text = Input(shape=(), dtype=tf.string, name="text")

# 2. Użycie opakowanej warstwy z TF-Hub
embedding = HubTextEmbedding("https://tfhub.dev/google/nnlm-en-dim50/2")(input_text)

# 3. Dalsze warstwy
x = Dense(16, activation='relu')(embedding)
output = Dense(1)(x)

# 4. Model
model = Model(inputs=input_text, outputs=output)

# 5. Kompilacja
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 6. Podsumowanie
model.summary()
