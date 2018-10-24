import keras
from .load_process_data import (
    ProcessedData
)


# Activation function tests
def cnn_relu():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(ProcessedData.num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def cnn_sigmoid():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='sigmoid'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='sigmoid'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='sigmoid'))
    model.add(keras.layers.Dense(50, activation='sigmoid'))
    model.add(keras.layers.Dense(ProcessedData.num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def cnn_elu():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='elu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='elu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='elu'))
    model.add(keras.layers.Dense(50, activation='elu'))
    model.add(keras.layers.Dense(ProcessedData.num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def cnn_mix():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='elu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='sigmoid'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(ProcessedData.num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
