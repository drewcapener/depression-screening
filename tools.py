import pandas as pd
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.dummy import DummyClassifier as dc
import numpy as np

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
        newThings.sort(key= lambda z : z[0])
        best = newThings[-1]
        print("new best = " + str(best[0]))
        print("using these features: " + str(best[1]))


data_frame = pd.read_csv("matthew_delta.csv")

data = data_frame.iloc[:, 1:]
data = data.dropna(how='any')
y = data.PHQ2.to_numpy()
X = data.drop("PHQ2", axis=1).to_numpy()

finalTrainData = np.empty((0,len(X[0])))
finalTrainLabels = np.empty((0,1))
finalTestData = np.empty((0,len(X[0])))
finalTestLabels = np.empty((0,1))

# For all the possible PHQ2 Delta values, get 70% of them for training and 30% for testing.
# Except on 0, then get 5% for training.
for x in range(-6,7):
    split = .05 if x == 0  else .7
    niceList = np.nonzero(y == x)
    newX = X[niceList]
    newY = np.reshape(y[niceList], (-1,1))
    newTrainData, newTrainLabels, newTestData, newTestLabels = splitData(newX, newY, split)
    finalTrainData = np.concatenate((finalTrainData, newTrainData))
    finalTrainLabels = np.concatenate((finalTrainLabels, newTrainLabels))
    finalTestData = np.concatenate((finalTestData, newTestData))
    finalTestLabels = np.concatenate((finalTestLabels, newTestLabels))


# Get a baseline accuracy:
dummyClassifier = dc(strategy="most_frequent")
dummyClassifier.fit(finalTrainData, finalTrainLabels)
print("Baseline accuracy: ")
print(dummyClassifier.score(finalTestData,finalTestLabels))

# Pick your classifier here and pass it in to the wrapper
mlp = mlp(max_iter=1000)
wrapper(mlp, finalTrainData, finalTrainLabels.flatten(), finalTestData, finalTestLabels.flatten())


