"""
Uses one-vs-one SVM to train a model that estimates age
from 
Authors: Kenny, Raymond, Rick
Date: 5/1/2019
"""
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn import svm  
from sklearn.model_selection import StratifiedKFold, GridSearchCV

class SVM:
    
    def __init__(self):
        """
        Initializes the SVM classifier
        """
        self.clf = svm.LinearSVC()

    def trainSVM(self, XTrain, yTrain):
        """  
        Purpose - trains the SVM classifier
        Param - XTrain - training data
                yTrain - training labels
        Return - clf - the trained classifier
        """
        # Train the classifier on the training data
        print("Training...")
        self.clf.fit(XTrain, yTrain)
        print("Done training.\n")
       
    def testSVM(self, XTest):
        """
        Purpose - runs the NB model on the test data
        Param - clf - trained clasifier
                XTest - test data
        Return - yPred - list of predicted labels
        """
        
        #creates list of predicted labels
        yPred = self.clf.predict(XTest)

        return yPred

    def evaluate(self, XTest, yTrue, yPred):
        """
        Purpose - gives us our test accuracy and confusion matrix
        Param - yTrue - list of actual test labels
                yPred - list of predicted test labels
        """

        # calculates the accuracy score on the test data
        svmScore = self.clf.score(XTest, yTrue)
        print("SVM classifer score was ", svmScore)

        # prints the confusion matrix
        # now call confusion matrix method
        print("\nConfusion Matrix")
        print(confusion_matrix(yTrue, yPred))
        
        return
