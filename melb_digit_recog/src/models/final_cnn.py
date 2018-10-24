import numpy as np
import keras
from .load_process_data import (
    ProcessedData
)

# Final CNN
np.random.seed(1994)


def cnn_model_final():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(45, (7, 7), input_shape=(1, 64, 64), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(ProcessedData.num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
