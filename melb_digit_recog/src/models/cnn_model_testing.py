import keras
from .load_process_data import (
    num_classes,
    n_train_X,
    n_train_y,
    n_test_X,
    n_test_y,
)


# CNN variable testing
def cnn_opt():
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


def cnn_conv():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(45, (7, 7), input_shape=(1, 64, 64), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_dense():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_drop():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.30))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Factor Testing
# models
cnn_drop = cnn_drop()
cnn_dense = cnn_dense()
cnn_conv = cnn_conv()
cnn_opt = cnn_opt()

cnn_drop.fit(n_train_X, n_train_y,
             validation_data=(n_test_X, n_test_y),
             epochs=5, batch_size=200, verbose=2)
score = cnn_drop.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

cnn_opt.fit(n_train_X, n_train_y,
            validation_data=(n_test_X, n_test_y),
            epochs=5, batch_size=200, verbose=2)
score = cnn_opt.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

cnn_dense.fit(n_train_X, n_train_y,
              validation_data=(n_test_X, n_test_y),
              epochs=5, batch_size=200, verbose=2)
score = cnn_dense.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

cnn_conv.fit(n_train_X, n_train_y,
             validation_data=(n_test_X, n_test_y),
             epochs=5, batch_size=200, verbose=2)
score = cnn_conv.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))
