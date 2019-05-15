"""
Uses multinomial Naive Bayes to train a model
Authors: Kenny, Raymond, Rick
Date: 5/1/2019
"""

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from collections import Counter

class NaiveBayes:

    def __init__(self):
        self.clf = MultinomialNB

    def trainNB(self, trainX, trainy):
        """
        Purpose - trains a Naive Bayes classifier
        Param - trainX - training data
                trainy - training labels
        """

        #initializes the Naive Bayes classifier
        self.clf = MultinomialNB()

        #uses the training data to train the classifier
        self.clf.fit(trainX, trainy)


    def testNB(self, testX):
        """
        Purpose - runs the NB model on the test data
        Param - testX - test data
        Return - yPred - list of predicted labels
        """

        #creates list of predicted labels
        yPred = self.clf.predict(testX)

        return yPred

    def evaluate(self, yTrue, yPred):
        """
        Purpose - gives us our test accuracy and confusion matrix
        Param - yTrue - list of actual test labels
                yPred - list of predicted test labels
        """

        #calculates the accuracy using Naive Bayes
        print("\nAccuracy using Naive Bayes")
        print(accuracy_score(yTrue, yPred))

        #prints the confusion matrix
        print("\nConfusion Matrix")
        print(confusion_matrix(yTrue,yPred))

        return
