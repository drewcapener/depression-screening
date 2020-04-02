import numpy as np
import pandas as pd


def splitData(X, y, trainingPercentage):
    combinedArray = np.concatenate((X, y), axis=1)
    np.random.shuffle(combinedArray)

    seventyPercent = (len(X)*(trainingPercentage*10))//10

    newX = combinedArray[:, :len(X[0])]
    newY = combinedArray[:, len(X[0]):]

    return newX[:int(seventyPercent)], newY[:int(seventyPercent)], newX[int(seventyPercent):], newY[int(seventyPercent):]


def wrapper(classifier, trainData, trainLabels, testData, testLabels):
    classifier.fit(trainData, trainLabels)
    print("Accuracy = " + str(classifier.score(testData, testLabels)))
    best = (classifier.score(testData, testLabels), np.arange(len(trainData[0])))
    while 1:
        newThings = []
        for x in range(len(best[1])):
            lst = np.delete(best[1], x)
            newTrainData = trainData[:,lst]
            newTestData = testData[:,lst]
            classifier.fit(newTrainData, trainLabels)
            newThings.append((classifier.score(newTestData, testLabels), lst))
        newThings.sort(key=lambda z: z[0])
        best = newThings[-1]
        print("new best = " + str(best[0]))
        print("using these features: " + str(best[1]))


def extract_data(path):
    data_frame = pd.read_csv(path)

    data = data_frame.iloc[:, 1:]
    data = data.dropna(how='any')
    y = data.PHQ2.to_numpy()
    X = data.drop("PHQ2", axis=1).to_numpy()
    return X, y


def split_data(X, y, training_percent=0.7, percent_zeroes=0.05):
    if percent_zeroes is not None and percent_zeroes > 0:
        # Get <percent_zeroes> percent of the data where y = 0 for training, the rest goes to test set
        permutation = np.random.permutation(np.where(y == 0)[0])
        split_idx = int(len(permutation) * percent_zeroes)
        zeroes_X_train, zeroes_y_train, zeroes_X_test, zeroes_y_test = apply_permutation(X, y, permutation, split_idx)

        # Get <training_percent> percent of the data where y != 0 for training, the rest goes to test set
        permutation = np.random.permutation(np.where(y != 0)[0])
        split_idx = int(len(permutation) * training_percent)
        nonzero_X_train, nonzero_y_train, nonzero_X_test, nonzero_y_test = apply_permutation(X, y, permutation, split_idx)

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
