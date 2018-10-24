import numpy as np
import keras
import pandas as pd

from .load_process_data import (
    ProcessedData
)


def cnn_model():
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
    model.add(keras.layers.Dense(ProcessedData.num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Running model
np.random.seed(1994)
num_pixels = 64 * 64
firstCnnModel = cnn_model()
firstCnnModel.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
                  validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
                  epochs=10, batch_size=200, verbose=2)
firstCnnModel_pred = firstCnnModel.predict(ProcessedData.n_test_X)
score = firstCnnModel.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
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


