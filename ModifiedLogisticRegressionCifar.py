from __future__ import division
from sklearn.metrics import accuracy_score, precision_score, recall_score

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
        self.trainingDataSet = trainingDataSet
        self.testingDataSet = testingDataSet
        self.learningRate = learningRate
        self.tolerance = tolerance
        self.maxIterations = maxIterations

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
        theta, iterations = self.logisticRegression(self.trainingDataSet)
        print("\tIterations: {}".format(iterations))
        trainAccuracy, trainPrecision, trainRecall = self.accuracy(
            self.trainingDataSet, theta)
        print("\tAccuracy: {}\n\tPrecision: {}\n\tRecall: {}".format(
            trainAccuracy, trainPrecision, trainRecall))

        print("Test Data Set:")
        testAccuracy, testPrecision, testRecall = self.accuracy(
            self.testingDataSet, theta)

        print("\tAccuracy:{}\n\tPrecision:{}\n\tRecall:{}".format(
            testAccuracy, testPrecision, testRecall))


def importData(fileName):
    dataSet = pd.read_pickle(fileName)

    return dataSet


if __name__ == "__main__":
    cifarTrainFileName = 'TrainData.pkl'
    cifarTrainDataSet = importData(cifarTrainFileName)

    cifarTestFileName = 'TestData.pkl'
    cifarTestDataSet = importData(cifarTestFileName)

    for learningRate in [0.1]:
        print("Learning Rate: {}".format(learningRate))
        for tolerance in [0.0001]:
            print("Tolerance: {}".format(tolerance))
            cifarLogisticRegression = CifarLogisticRegression(
                cifarTrainDataSet, cifarTestDataSet, learningRate, tolerance)
            cifarLogisticRegression.validate()
