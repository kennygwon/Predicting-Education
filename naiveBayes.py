"""
Uses multinomial Naive Bayes to train a model
Authors: Kenny, Raymond, Rick
Date: 5/1/2019
"""

class NaiveBayes:
    
    def __init__(self):
        pass

    def trainNB(self, trainX, trainy):
        """  
        Purpose - trains a Naive Bayes classifier
        Param - trainX - training data
                trainy - training labels
        Return - clf - the trained classifier
        """

        #initializes the Naive Bayes classifier
        clf = MultinomialNB()

        #uses the training data to train the classifier
        clf.fit(trainX, trainy)
       
        return clf

    def testNB(self, testX):
        """
        Purpose - runs the NB model on the test data
        Param - clf - trained clasifier
                testX - test data
        Return - yPred - list of predicted labels
        """
        
        #creates list of predicted labels
        yPred = clf.predict(testX)

        return yPred

    def evaluate(self, trainy, yTrue, yPred):
        """
        Purpose - gives us our test accuracy and confusion matrix
        Param - yTrue - list of actual test labels
                yPred - list of predicted test labels
        """

        #calculates the accuracy using MFS
        counterTrain = Counter(trainy)
        maxCount = max(counterTrain)
        for key in list(counterTrain.keys()):
          if counterTrain[key] == maxCount:
              MFS = key
        counterTest = Counter(yTrue)
        print("\nAccuracy using MFS")
        print(counterTest[MFS] / len(yTrue))

        #calculates the accuracy using Naive Bayes
        print("\nAccuracy using Naive Bayes")
        print(accuracy_score(yTrue, yPred))

        #prints the confusion matrix
        print("\nConfusion Matrix")
        print(confusion_matrix(yTrue,yPred))
        
        return
