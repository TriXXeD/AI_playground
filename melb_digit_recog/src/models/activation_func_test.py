import keras
from .load_process_data import (
    num_classes,
    n_train_X,
    n_train_y,
    n_test_X,
    n_test_y,
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
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
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
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
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
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
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
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# Running models
sigmoid = cnn_sigmoid()
relu = cnn_relu()
elu = cnn_elu()
mix = cnn_mix()

relu.fit(n_train_X, n_train_y,
         validation_data=(n_test_X, n_test_y),
         epochs=6, batch_size=200, verbose=2)
score = relu.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

sigmoid.fit(n_train_X, n_train_y,
            validation_data=(n_test_X, n_test_y),
            epochs=6, batch_size=200, verbose=2)
score = sigmoid.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

elu.fit(n_train_X, n_train_y,
        validation_data=(n_test_X, n_test_y),
        epochs=6, batch_size=200, verbose=2)
score = elu.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

mix.fit(n_train_X, n_train_y,
        validation_data=(n_test_X, n_test_y),
        epochs=6, batch_size=200, verbose=2)
score = mix.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))
