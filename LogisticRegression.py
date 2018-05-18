from __future__ import division
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self,
                 dataSet,
                 learningRate=0.001,
                 tolerance=0.001,
                 maxIterations=1000,
                 kFold=10):
        self.dataSet = dataSet.fillna(0)
        self.learningRate = learningRate
        self.tolerance = tolerance
        self.maxIterations = maxIterations

        self.kFold = kFold
        self.kf = KFold(n_splits=kFold, shuffle=True)

    def zScoreNormalization(self, dataSet, attributeMeans=[],
                            attributeStds=[]):
        normalizedDataSet = dataSet.copy(True)
        attributes = dataSet.shape[1] - 1

        if len(attributeMeans) == attributes:
            for i in range(attributes):
                attributeMean = attributeMeans[i]
                attributeStd = attributeStds[i]
                normalizedDataSet[i] = normalizedDataSet[i].apply(
                    lambda x: (x - attributeMean) / attributeStd if attributeStd > 0 else 0
                )

            return normalizedDataSet, attributeMeans, attributeStds
        else:
            newAttributeMeans = []
            newAttributeStds = []

            for i in range(attributes):
                attributeMean = dataSet[i].mean()
                newAttributeMeans.append(attributeMean)
                attributeStd = dataSet[i].std()
                newAttributeStds.append(attributeStd)
                normalizedDataSet[i] = normalizedDataSet[i].apply(
                    lambda x: (x - attributeMean) / attributeStd if attributeStd > 0 else 0
                )

            return normalizedDataSet, newAttributeMeans, newAttributeStds

    def constantFeature(self, dataSet):
        regressionDataSet = dataSet.copy(True)

        regressionDataSet.columns = range(1, regressionDataSet.shape[1] + 1)
        regressionDataSet.insert(0, 0, 1)

        return regressionDataSet

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def hypothesis(self, X, theta):
        return self.sigmoid(np.dot(X, theta))

    def cost(self, X, Y, theta):
        probabilities = self.hypothesis(X, theta)

        return (np.multiply(-Y, np.log(probabilities)) - np.multiply(
            (1 - Y), np.log(1 - probabilities))).mean()

    def gradient(self, X, Y, theta):
        prediction = self.hypothesis(X, theta)
        error = prediction - Y

        return np.dot(X.T, error) / X.shape[0]

    def logisticRegression(self, dataSet, plotGraph):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = np.matrix(dataSet.iloc[:, -1]).T

        self.attributes = dataSet.shape[1] - 1
        theta = np.matrix(np.zeros(self.attributes)).T

        logisticLoss = [self.cost(X, Y, theta)]
        for i in range(self.maxIterations):
            gradient = self.gradient(X, Y, theta)
            newTheta = theta - self.learningRate * gradient
            newCost = self.cost(X, Y, newTheta)
            if (logisticLoss[i] - newCost < self.tolerance):
                break
            else:
                logisticLoss.append(newCost)
                theta = newTheta

        if plotGraph:
            plt.plot(logisticLoss)
            plt.xlabel('Iteration')
            plt.ylabel('Logistic Loss')
            plt.title('Logistic Regression')
            plt.show()

        return theta

    def predict(self, X, theta):
        probabilities = self.hypothesis(X, theta)

        return [
            1 if probability >= 0.5 else 0 for probability in probabilities
        ]

    def accuracy(self, dataSet, theta):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = dataSet.iloc[:, -1]

        prediction = self.predict(X, theta)

        accuracy = accuracy_score(Y, prediction)
        precision = precision_score(Y, prediction, average="weighted")
        recall = recall_score(Y, prediction, average="weighted")

        return accuracy, precision, recall

    def validate(self):
        trainAccuracies = []
        trainPrecisions = []
        trainRecalls = []
        testAccuracies = []
        testPrecisions = []
        testRecalls = []

        fold = 1
        plotFold = random.randint(1, self.kFold + 1)

        print(
            "Fold\tTraining Accuracy\tTraining Precision\tTraining Recall\tTest Accuracy\tTest Precision\tTest Recall"
        )
        for trainIndex, testIndex in self.kf.split(self.dataSet):
            trainDataSet, trainAttributeMeans, trainAttributeStds = self.zScoreNormalization(
                self.dataSet.iloc[trainIndex])
            trainDataSet = self.constantFeature(trainDataSet)
            theta = self.logisticRegression(trainDataSet, fold == plotFold)
            trainAccuracy, trainPrecision, trainRecall = self.accuracy(
                trainDataSet, theta)
            trainAccuracies.append(trainAccuracy)
            trainPrecisions.append(trainPrecision)
            trainRecalls.append(trainRecall)

            testDataSet, _, _ = self.zScoreNormalization(
                self.dataSet.iloc[testIndex], trainAttributeMeans,
                trainAttributeStds)
            testDataSet = self.constantFeature(testDataSet)
            testAccuracy, testPrecision, testRecall = self.accuracy(
                testDataSet, theta)
            testAccuracies.append(testAccuracy)
            testPrecisions.append(testPrecision)
            testRecalls.append(testRecall)

            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                fold, trainAccuracy, trainPrecision, trainRecall, testAccuracy,
                testPrecision, testRecall))
            fold += 1

        print("Mean\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.mean(trainAccuracies), np.mean(trainPrecisions),
            np.mean(trainRecalls), np.mean(testAccuracies),
            np.mean(testPrecisions), np.mean(testRecalls)))
        print("Standard Deviation\t{}\t{}\t{}\t{}\t{}\t{}".format(
            np.std(trainAccuracies), np.std(trainPrecisions),
            np.std(trainRecalls), np.std(testAccuracies),
            np.std(testPrecisions), np.std(testRecalls)))


def importData(dataname):
    mnist = fetch_mldata(dataname)
    dataSet = pd.DataFrame(np.c_[mnist['data'], mnist['target']])
    dataSet = dataSet.drop_duplicates()
    label = dataSet.shape[1] - 1
    dataSet[label] = dataSet[label].apply(lambda x: 0.0 if x > 1.0 else x)

    return dataSet


if __name__ == "__main__":
    mnistDataname = 'MNIST original'
    mnistDataSet = importData(mnistDataname)
    mnistLogisticRegression = LogisticRegression(mnistDataSet, 0.1, 0.001)
    mnistLogisticRegression.validate()
