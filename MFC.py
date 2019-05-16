"""
Uses Most Frequent Class to obtain accuracy
Authors: Kenny, Raymond, Rick
Date: 5/1/2019
"""

from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

def evaluate(trainy, yTrue):
    """
    Purpose - calculates accuracy and prints confusion matrix
    Param - trainy - list of actual train labels
            yTrue - list of actual test labels
    """

    #calculates the accuracy using MFS
    counterTrain = Counter(trainy)
    maxCount = max(counterTrain.values())
    for key in list(counterTrain.keys()):
      if counterTrain[key] == maxCount:
          MFS = key
    counterTest = Counter(yTrue)
    print("\nAccuracy using MFS")
    accuracy = counterTest[MFS] / len(yTrue)
    print(accuracy)

    #creates the list of predicted values
    yPred = [MFS] * len(yTrue)

    #prints the confusion matrix
    print("\nConfusion Matrix")
    print(confusion_matrix(yTrue,yPred))

    return accuracy
