import numpy as np
from keras.utils import np_utils
from scipy.signal import argrelextrema

# Load Data
k_data = np.load('../data/data.npz')
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


class MultiDigitData:
    for i in range(500):
        r1 = np.random.randint(0, 34999)
        r2 = np.random.randint(0, 34999)
        r3 = np.random.randint(0, 34999)
        # Removing extra dimension for stacking, test to see if dimension is required for model
        new_train = k_train_X.reshape(k_train_X.shape[0], 64, 64)
        new_train1 = np.array(new_train[r1])
        new_train2 = np.array(new_train[r2])
        new_train3 = np.array(new_train[r3])

        two_digits = np.vstack((new_train1, new_train2))
        three_digits = np.vstack((new_train1, new_train2, new_train3))
        bin_two_digits = two_digits / 255
        bin_three_digits = three_digits / 255

        two_column_sums = bin_two_digits.sum(axis=1)
        two_median = np.median(two_column_sums)
        two_avg = np.average(two_column_sums)
        three_column_sums = bin_three_digits.sum(axis=1)
        three_median = np.median(three_column_sums)
        three_avg = np.average(three_column_sums)
        local_max_two = argrelextrema(two_column_sums, np.greater)

        # Works if two_column is global variable, need to bring into list comp scope
        # local_max_over_median_two = [x for x in local_max_two[0] if two_column_sums[x] > two_median]
        local_max_over_median_two = []
        for x in local_max_two[0]:
            if two_column_sums[x] > (two_digits.shape[1] * 0.85):
                local_max_over_median_two.append(x)

        print(local_max_over_median_two)
        local_max_three = argrelextrema(three_column_sums, np.greater)
        # print(two_column_sums, three_column_sums)
        print("two digits: {} {} \nthree digits: {} {}".format(two_median, two_avg, three_median, three_avg))
        print(local_max_two, local_max_over_median_two)
        print(two_column_sums[local_max_over_median_two])


class ProcessedData:
    # normalize
    n_train_X = k_train_X / 255
    n_test_X = k_test_X / 255

    # one hot encoding
    n_train_y = np_utils.to_categorical(k_train_y)
    n_test_y = np_utils.to_categorical(k_test_y)

    num_classes = n_test_y.shape[1]
