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
        newThings.sort(key=lambda z: z[0])
        best = newThings[-1]
        print("new best = " + str(best[0]))
        print("using these features: " + str(best[1]))
