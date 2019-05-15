"""
Uses one-vs-one SVM to train a model that estimates age
from 
Authors: Kenny, Raymond, Rick
Date: 5/1/2019
"""
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn import svm  
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

class SVM:
    
    def __init__(self):
        """
        Initializes the SVM classifier
        """
        self.clf = svm.LinearSVC(C=0.1, random_state=42)
        self.svc_test_scores = None

    def runTuneTest(self, params, X, y):
        """
        Purpose - Tunes the SVC classifiers hyper-parameters. 
                  This method handles creation of train/tune/test sets and runs
                  the pipeline, then reports the scores of using various sets 
                  of hyperparameters. We are using 5 folds here.
        Params -  params: a dictionary of hyperparameters for GridSearchCV
                  X: the training data
                  y: the training labels
        Returns - The scores using each distinct hyperparameter value.
        """

        print("\nTuning SVC...\n")
        fold_num = 1
        test_scores = []
        stratifier = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in stratifier.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            # change y into an np array
            y =np.array(y)
            y_train, y_test = y[train_index], y[test_index]
            clf = GridSearchCV(self.clf, params)
            clf.fit(X_train, y_train)
            score = clf.score(X_train, y_train)
            print("Fold number: ", fold_num)
            print("Best Parameter: ", clf.best_params_)
            print("Training score: ", score, "\n")
            test_scores.append(clf.score(X_test, y_test))
            fold_num += 1

        self.svc_test_scores = test_scores
    
    def printTestScores(self):
        """
        Purpose - prints out the scores from the tune test
        Params -  none
        Returns - nothing, but has the side effect of printing the 
                  tune test scores for the different folds
        """
        print("---------------------------------------------")
        print("Fold    SVM Test Accuracy")
        for i in range(5):
            print("%4d %19.3f" % (i, self.svc_test_scores[i]))
        print("\n")

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

    def visualizeWeights(self, features):
        """
        Purpose - draws bar graphs that help analyse feature importance using 
                  corresponding feature weights from our trained classifier.
                  We have 32 features in total, but we will analyse the top 10
                  most important features 
        Params -  features: a list of length p containing the feature names for 
                  our SVM classifier  
        Returns - Nothing, prints two graph using their absolute weight values
        """
        weight_mtx = self.clf.coef_.tolist() 
        y_pos = np.arange(10) 

        # NB: We have seven models, as Linear SVC applies a One Vs All criterion for
        # multiclass classification. We shall compute the top 5 features for each 
        # model
        
        k = 1
        for weight_vect in weight_mtx:
            # create a dictionaries to keep track of indices while sorting
            weight_dict = {}

            for i in range(len(weight_vect)):
                weight_dict[features[i]] = weight_vect[i]

            sorted_weights = sorted(weight_dict.items(), key=lambda kv: kv[1], reverse=True)

            print("Model #: ", k)
            print("Top 5 features are : ")
            for j in range(5):
                print(sorted_weights[j])
            print("\n--------------------------------------------------------")
            k += 1 

            top_features = [pair[0] for pair in sorted_weights]
            top_weights = [pair[1] for pair in sorted_weights]

            top_10_features = top_features[:10]
            top_10_weights = top_weights[:10]

            plt.bar(y_pos, top_10_weights,  align='center', width=0.5)
            plt.xticks(y_pos, top_10_features)
            plt.ylabel("Weight Value")
            plt.title("SVM coefficient analysis using regular weights")
            plt.show()
            
