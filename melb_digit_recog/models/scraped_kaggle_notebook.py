import numpy as np
import keras
import pandas as pd
from keras import backend as K
from keras.utils import np_utils

np.random.seed(1994)

# Kaggle competetion Code
# Load Data
k_data = np.load('data.npz')
keys = k_data.keys()
print(keys)
k_train_X = k_data['train_X']
k_train_y = k_data['train_y']
k_test_X = k_data['test_X']

print(k_train_X.shape)
print(k_train_y.shape)
print(k_test_X.shape)

# load test y data
k_test_y = np.zeros([500, ])
k_test_y[:, ] = ([
    9, 3, 6, 9, 7, 8, 6, 7, 1, 6, 6, 3, 0, 8, 6, 4, 3, 1, 2, 5, 0, 5, 0, 3, 0, 8, 4, 7,
    9, 5, 1, 7, 1, 7, 6, 0, 1, 3, 4, 2, 3, 1, 3, 6, 4, 0, 2, 3, 2, 7, 5, 4, 5, 2, 2, 4,
    9, 7, 9, 3, 3, 4, 6, 5, 6, 7, 7, 6, 1, 0, 9, 6, 3, 3, 0, 6, 0, 7, 6, 3, 1, 7, 4, 9,
    9, 7, 6, 4, 2, 4, 1, 3, 1, 3, 9, 5, 7, 8, 5, 0, 2, 8, 6, 1, 6, 0, 0, 6, 5, 5, 4, 8,
    8, 1, 3, 0, 1, 3, 8, 4, 2, 1, 7, 8, 5, 1, 0, 1, 4, 2, 8, 6, 3, 4, 9, 4, 4, 8, 7, 0,
    3, 6, 5, 3, 8, 4, 1, 7, 3, 1, 8, 2, 0, 5, 3, 8, 3, 0, 2, 0, 1, 7, 8, 8, 4, 5, 5, 7,
    1, 6, 4, 3, 0, 0, 8, 5, 9, 4, 9, 4, 8, 2, 0, 5, 8, 1, 0, 6, 7, 2, 9, 0, 2, 9, 3, 6,
    1, 7, 2, 7, 0, 3, 6, 2, 6, 4, 8, 9, 7, 9, 3, 9, 0, 7, 1, 2, 3, 3, 5, 6, 4, 7, 0, 8,
    4, 7, 4, 5, 3, 5, 1, 4, 7, 0, 8, 5, 6, 2, 1, 4, 7, 3, 2, 9, 8, 8, 0, 2, 9, 6, 8, 2,
    4, 8, 1, 4, 1, 1, 6, 6, 3, 8, 8, 0, 4, 6, 2, 7, 2, 9, 8, 9, 5, 7, 5, 5, 2, 8, 4, 9,
    1, 5, 0, 9, 2, 3, 2, 0, 3, 9, 6, 8, 5, 7, 8, 1, 4, 3, 9, 2, 3, 4, 9, 4, 1, 9, 0, 2,
    8, 1, 0, 2, 7, 0, 8, 0, 9, 2, 1, 4, 6, 8, 5, 4, 5, 8, 3, 9, 6, 9, 0, 4, 5, 9, 5, 8,
    5, 5, 7, 1, 7, 4, 7, 3, 3, 5, 3, 7, 5, 0, 2, 1, 4, 4, 4, 4, 3, 9, 7, 2, 9, 4, 2, 3,
    7, 6, 9, 4, 0, 5, 9, 3, 6, 7, 0, 0, 1, 4, 7, 9, 8, 8, 0, 5, 6, 0, 5, 6, 1, 9, 3, 1,
    0, 4, 7, 6, 4, 9, 9, 8, 6, 1, 9, 7, 2, 2, 8, 2, 6, 6, 3, 5, 0, 3, 2, 8, 9, 1, 1, 5,
    2, 7, 5, 6, 5, 0, 6, 5, 7, 8, 5, 9, 8, 8, 1, 7, 1, 8, 3, 2, 1, 5, 0, 1, 9, 5, 8, 8,
    5, 9, 2, 1, 3, 2, 2, 2, 2, 9, 0, 1, 2, 5, 8, 7, 2, 6, 2, 6, 7, 9, 1, 0, 6, 0, 8, 2,
    7, 4, 3, 9, 3, 0, 9, 4, 5, 6, 6, 8, 6, 1, 1, 8, 6, 7, 5, 8, 1, 4, 1, 5
])

# Pre-processing

# cnn prep
k_train_X = k_train_X.reshape(k_train_X.shape[0], 1, 64, 64).astype('float32')
k_test_X = k_test_X.reshape(k_test_X.shape[0], 1, 64, 64).astype('float32')

# normalize
n_train_X = k_train_X / 255
n_test_X = k_test_X / 255
# one hot encoding

n_train_y = np_utils.to_categorical(k_train_y)
n_test_y = np_utils.to_categorical(k_test_y)
num_classes = n_test_y.shape[1]

# Keras - Using Tensorflow backend
# Model setup

K.set_image_dim_ordering('th')


# Model creation


def cnnModel():
    # Model layers and functions
    model = keras.models.Sequential()
    model.add(keras.layers.convolutional.Conv2D(30, (5, 5), input_shape=(1, 64, 64), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.convolutional.Conv2D(15, (3, 3), activation='relu'))
    model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Running model
np.random.seed(1994)
num_pixels = 64 * 64
firstCnnModel = cnnModel()
firstCnnModel.fit(n_train_X, n_train_y,
                  validation_data=(n_test_X, n_test_y),
                  epochs=10, batch_size=200, verbose=2)
firstCnnModel_pred = firstCnnModel.predict(n_test_X)
score = firstCnnModel.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.2f%%" % (100 - score[1] * 100))

# Reversing categorical data for submission
csv_file = np.zeros([500, 2])
for i in range(firstCnnModel_pred.shape[0]):
    csv_file[i] = ([i + 1, np.argmax(firstCnnModel_pred[i])])

# print(csv_file)

labels = (['Id', 'Label'])
csv_int = csv_file.astype(int)
handin = np.vstack([labels, csv_int])

df = pd.DataFrame(handin)
df.to_csv("pred1.csv", header=None, index=None, index_label=None)


# CNN variable testing


def cnnOpt():
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


def cnnConv():
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


def cnnDense():
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


def cnnDrop():
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
cnn_drop = cnnDrop()
cnn_dense = cnnDense()
cnn_conv = cnnConv()
cnn_opt = cnnOpt()

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


# Activation function tests
def cnnRelu():
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


def cnnSigmoid():
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


def cnnELU():
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


def cnnMix():
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
sigmoid = cnnSigmoid()
relu = cnnRelu()
elu = cnnELU()
mix = cnnMix()

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

# Final CNN
np.random.seed(1994)


def cnnModelFinal():
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
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# the final run
cnnFinal = cnnModelFinal()

np.random.seed(1994)
cnnFinal.fit(n_train_X, n_train_y,
             validation_data=(n_test_X, n_test_y),
             epochs=15, batch_size=200, verbose=2)
score = cnnFinal.evaluate(n_test_X, n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))


final_pred = cnnFinal.predict(n_test_X)

# Reversing categorical data for submission
csv_final = np.zeros([500, 2])
for i in range(final_pred.shape[0]):
    csv_final[i] = ([i+1, np.argmax(final_pred[i])])

# print(csv_file)
labels = (['Id', 'Label'])
csv_final_int = csv_final.astype(int)
final_handin = np.vstack([labels, csv_final_int])

df = pd.DataFrame(final_handin)
df.to_csv("final_prediction.csv", header=None, index=None, index_label=None)
