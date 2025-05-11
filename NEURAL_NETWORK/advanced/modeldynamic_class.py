import tensorflow as tf
from tensorflow.keras import layers, models
import json

class DynamicNNBuilder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.layer_config = []
    
    def add_conv(self, filters, kernel_size, activation='relu', pool_size=2):
        self.layer_config.append({
            'type': 'conv',
            'filters': filters,
            'kernel_size': kernel_size,
            'activation': activation,
            'pool_size': pool_size
        })
    
    def add_dense(self, units, activation='relu', dropout=None):
        self.layer_config.append({
            'type': 'dense',
            'units': units,
            'activation': activation,
            'dropout': dropout
        })
    
    def add_flatten(self):
        self.layer_config.append({'type': 'flatten'})
    
    def add_lstm(self, units, return_sequences=False):
        self.layer_config.append({
            'type': 'lstm',
            'units': units,
            'return_sequences': return_sequences
        })
    
    def build_model(self, output_units, output_activation='softmax', optimizer='adam'):
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))
        
        for cfg in self.layer_config:
            if cfg['type'] == 'conv':
                model.add(layers.Conv2D(cfg['filters'], cfg['kernel_size'], activation=cfg['activation']))
                model.add(layers.MaxPooling2D(cfg['pool_size']))
            elif cfg['type'] == 'dense':
                model.add(layers.Dense(cfg['units'], activation=cfg['activation']))
                if cfg['dropout']:
                    model.add(layers.Dropout(cfg['dropout']))
            elif cfg['type'] == 'flatten':
                model.add(layers.Flatten())
            elif cfg['type'] == 'lstm':
                model.add(layers.LSTM(cfg['units'], return_sequences=cfg['return_sequences']))
        
        model.add(layers.Dense(output_units, activation=output_activation))
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def export_config(self):
        return json.dumps(self.layer_config, indent=2)
    
    def import_config(self, config_json):
        self.layer_config = json.loads(config_json)



builder = DynamicNNBuilder(input_shape=(28, 28, 1))
builder.add_conv(filters=32, kernel_size=3)
builder.add_conv(filters=64, kernel_size=3)
builder.add_flatten()
builder.add_dense(units=128, dropout=0.3)
model = builder.build_model(output_units=10)

model.summary()

json_config = builder.export_config()
print(json_config)

# Można później załadować:
builder2 = DynamicNNBuilder(input_shape=(28, 28, 1))
builder2.import_config(json_config)
model2 = builder2.build_model(output_units=10)
