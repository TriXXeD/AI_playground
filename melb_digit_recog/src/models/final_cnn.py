import numpy as np
import keras
import pandas as pd
from .load_process_data import (
    num_classes,
    n_train_X,
    n_train_y,
    n_test_X,
    n_test_y,
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
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    # Create model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


# the final run
cnnFinal = cnn_model_final()

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
