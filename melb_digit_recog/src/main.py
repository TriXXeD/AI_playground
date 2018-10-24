import numpy as np
import pandas as pd
from models.load_process_data import ProcessedData
from models.base_model import cnn_model
from models.cnn_model_testing import (
    cnn_opt,
    cnn_conv,
    cnn_dense,
    cnn_drop,
)
from models.activation_func_test import (
    cnn_elu,
    cnn_mix,
    cnn_relu,
    cnn_sigmoid,
)
from models.final_cnn import cnn_model_final

# Running base model
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

# Factor Testing

# Model variable testing
cnn_drop = cnn_drop()
cnn_dense = cnn_dense()
cnn_conv = cnn_conv()
cnn_opt = cnn_opt()

cnn_drop.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
             validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
             epochs=5, batch_size=200, verbose=2)
score = cnn_drop.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

cnn_opt.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
            validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
            epochs=5, batch_size=200, verbose=2)
score = cnn_opt.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

cnn_dense.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
              validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
              epochs=5, batch_size=200, verbose=2)
score = cnn_dense.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

cnn_conv.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
             validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
             epochs=5, batch_size=200, verbose=2)
score = cnn_conv.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

# Activation function comparison
sigmoid = cnn_sigmoid()
relu = cnn_relu()
elu = cnn_elu()
mix = cnn_mix()

relu.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
         validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
         epochs=6, batch_size=200, verbose=2)
score = relu.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

sigmoid.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
            validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
            epochs=6, batch_size=200, verbose=2)
score = sigmoid.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

elu.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
        validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
        epochs=6, batch_size=200, verbose=2)
score = elu.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

mix.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
        validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
        epochs=6, batch_size=200, verbose=2)
score = mix.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))

# Training and running final model
cnnFinal = cnn_model_final()

np.random.seed(1994)
cnnFinal.fit(ProcessedData.n_train_X, ProcessedData.n_train_y,
             validation_data=(ProcessedData.n_test_X, ProcessedData.n_test_y),
             epochs=15, batch_size=200, verbose=2)
score = cnnFinal.evaluate(ProcessedData.n_test_X, ProcessedData.n_test_y, verbose=2)
print("Error: %.3f%%" % (100 - score[1] * 100))


final_pred = cnnFinal.predict(ProcessedData.n_test_X)

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
