from __future__ import division
from sklearn.metrics import accuracy_score, precision_score, recall_score

import cPickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CifarLogisticRegression:
    def __init__(self,
                 trainingDataSet,
                 testingDataSet,
                 learningRate=0.001,
                 tolerance=0.001,
                 maxIterations=1000):
        self.trainingDataSet = trainingDataSet.fillna(0)
        self.testingDataSet = testingDataSet.fillna(0)
        self.learningRate = learningRate
        self.tolerance = tolerance
        self.maxIterations = maxIterations

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

    def logisticRegression(self, dataSet):
        X = np.matrix(dataSet.iloc[:, :-1])
        Y = np.matrix(dataSet.iloc[:, -1]).T

        attributes = dataSet.shape[1] - 1
        theta = np.matrix(np.random.uniform(-1, 1, attributes)).T

        logisticLoss = [self.cost(X, Y, theta)]
        iterations = 0
        for i in range(self.maxIterations):
            iterations = i + 1
            print("KAKA:: Iteration:{}".format(iterations))
            gradient = self.gradient(X, Y, theta)
            newTheta = theta - self.learningRate * gradient
            newCost = self.cost(X, Y, newTheta)
            if (logisticLoss[i] - newCost < self.tolerance):
                break
            else:
                logisticLoss.append(newCost)
                theta = newTheta

        plt.plot(logisticLoss)
        plt.xlabel('Iteration')
        plt.ylabel('Logistic Loss')
        plt.title('Logistic Regression')
        plt.show()

        return theta, iterations

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
        print("Training Data Set:")
        trainDataSet, trainAttributeMeans, trainAttributeStds = self.zScoreNormalization(
            self.trainingDataSet)
        print("KAKA:: trainDataSet Normalized")
        trainDataSet = self.constantFeature(trainDataSet)
        print("KAKA:: trainDataSet Constants Added")
        theta, iterations = self.logisticRegression(trainDataSet)
        print("\tIterations: {}".format(iterations))
        trainAccuracy, trainPrecision, trainRecall = self.accuracy(
            trainDataSet, theta)
        print("\tAccuracy: {}\n\tPrecision: {}\n\tRecall: {}".format(
            trainAccuracy, trainPrecision, trainRecall))

        print("Test Data Set:")
        testDataSet, _, _ = self.zScoreNormalization(
            self.testingDataSet, trainAttributeMeans, trainAttributeStds)
        print("KAKA:: testDataSet Normalized")
        testDataSet = self.constantFeature(testDataSet)
        print("KAKA:: testDataSet Constants Added")
        testAccuracy, testPrecision, testRecall = self.accuracy(
            testDataSet, theta)

        print("\tAccuracy:{}\n\tPrecision:{}\n\tRecall:{}".format(
            testAccuracy, testPrecision, testRecall))


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def importData(directoryName, fileNames):
    dataSets = []
    for file in fileNames:
        tempDict = unpickle("{}/{}".format(directoryName, file))
        dataSet = pd.DataFrame(np.c_[tempDict['data'], tempDict['labels']])
        dataSets.append(dataSet)
    dataSet = pd.concat(dataSets)

    label = dataSet.shape[1] - 1
    dataSet[label] = dataSet[label].apply(lambda x: 0.0 if x > 1.0 else x)

    return dataSet


if __name__ == "__main__":
    directoryName = '/Users/ledlab/Downloads/cifar-10-batches-py'

    cifarTrainFileNames = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
        'data_batch_5'
    ]
    cifarTrainDataSet = importData(directoryName, cifarTrainFileNames)
    print("KAKA:: Loaded Train Data")

    cifarTestFileNames = ['test_batch']
    cifarTestDataSet = importData(directoryName, cifarTestFileNames)
    print("KAKA:: Loaded Test Data")

    cifarLogisticRegression = CifarLogisticRegression(
        cifarTrainDataSet, cifarTestDataSet, 0.1, 0.000001)
    cifarLogisticRegression.validate()
