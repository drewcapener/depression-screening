import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

def wrapper(classifier, trainData, trainLabels, testData, testLabels):
    classifier.fit(trainData, trainLabels)
    print("Accuracy = " + str(score(classifier, testData, testLabels)))
    best = (score(classifier, testData, testLabels), np.arange(len(trainData[0])))
    while 1:
        newThings = []
        for x in range(len(best[1])):
            lst = np.delete(best[1], x)
            newTrainData = trainData[:,lst]
            newTestData = testData[:,lst]
            classifier.fit(newTrainData, trainLabels)
            newThings.append((score(classifier, newTestData, testLabels), lst))
        newThings.sort(key=lambda z: z[0])
        best = newThings[0]
        print("new best = " + str(best[0]))
        print("using these features: " + str(best[1]))

def score(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def extract_data(path, normalized=False):
    data_frame = pd.read_csv(path)
    data = data_frame.iloc[:, 1:]
    data = data.dropna(how='any')
    y = data.PHQ2.to_numpy()
    X = data.drop("PHQ2", axis=1).to_numpy()
    if normalized:
        X = preprocessing.minmax_scale(X)
        y = preprocessing.minmax_scale(y)
    return X, y


def split_data(X, y, training_percent=0.7, percent_zeroes=0.05, n_nonzero_repeat=0):
    if percent_zeroes is not None and percent_zeroes > 0:
        # Get <percent_zeroes> percent of the data where y = 0 for training, the rest goes to test set
        permutation = np.random.permutation(np.where(y == 0)[0])
        split_idx = int(len(permutation) * percent_zeroes)
        zeroes_X_train, zeroes_y_train, zeroes_X_test, zeroes_y_test = apply_permutation(X, y, permutation, split_idx)

        # Get <training_percent> percent of the data where y != 0 for training, the rest goes to test set
        permutation = np.random.permutation(np.where(y != 0)[0])
        split_idx = int(len(permutation) * training_percent)
        nonzero_X_train, nonzero_y_train, nonzero_X_test, nonzero_y_test = apply_permutation(X, y, permutation, split_idx)
        nonzero_X_train = np.repeat(nonzero_X_train, n_nonzero_repeat, axis=0)
        nonzero_y_train = np.repeat(nonzero_y_train, n_nonzero_repeat, axis=0)

        # Concatenate train/test sets with both zero/nonzero sets
        X_train, y_train = np.concatenate((zeroes_X_train, nonzero_X_train)), np.concatenate((zeroes_y_train, nonzero_y_train))
        X_test, y_test = np.concatenate((zeroes_X_test, nonzero_X_test)), np.concatenate((zeroes_y_test, nonzero_y_test))
        return X_train, y_train, X_test, y_test
    else:
        split_idx = int(len(X) * training_percent)
        permutation = np.random.permutation(np.arange(len(X)))
        return apply_permutation(X, y, permutation, split_idx)


def apply_permutation(X, y, permutation, split_idx):
    return X[permutation[:split_idx]], y[permutation[:split_idx]], \
           X[permutation[split_idx:]], y[permutation[split_idx:]]


def shuffle_data(X, y):
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]