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
        Param   - yTrue - list of actual test labels
                  yPred - list of predicted test labels
        Returns - svmScore: score of the SVM classifier
        """
        # calculates the accuracy score on  the test data
        svmScore = self.clf.score(XTest, yTrue)
        print("SVM classifer score was ", svmScore)
        # prints the confusion matrix
        # now call confusion matrix method
        print("\nConfusion Matrix")
        print(confusion_matrix(yTrue, yPred))

        return svmScore

    def visualizeWeights(self, features, feat_means):
        """
        Purpose - draws bar graphs that help analyse feature importance using 
                  corresponding feature weights from our trained classifier.
                  We have 32 features in total, but we will analyse the top 10
                  most important features 
        Params -  features: a list of length p containing the feature names for 
                  our SVM classifier  
        Returns - Nothing, prints top features for each model and displays a
                  graphical representation for each
        """
        weights = self.clf.coef_ 
        y_pos = np.arange(5) 
        
        
        # calculate (w*x) to see relative impact of weight on class means 
        wT_x = np.multiply(weights, feat_means)
        wT_x = np.reshape(wT_x, (87,))

        feat_impact_dict = {}
        original_vals = {}
        for i in range(len(features)):
            # use absolute values for sorting
            feat_impact_dict[features[i]] = abs(wT_x[i]) 
            # save original values for later analysis
            original_vals[features[i]] = wT_x[i]

        sorted_feat_impacts = sorted(feat_impact_dict.items(), key=lambda kv: kv[1], reverse=True)

        top_features = [pair[0] for pair in sorted_feat_impacts]
        top_5_features = top_features[:5]
        top_5_wTx = []

        print("\nHere are the top 5 most impactful features (SVM, bin. classification):\n")
        for feature in top_5_features:
            original_impact_val = original_vals[feature]
            top_5_wTx.append(original_impact_val)
            print("%s : %f" % (feature, original_impact_val))
        print("---------------------------------------------------\n")    

        # graphically,
        plt.bar(y_pos, top_5_wTx,  align='center', width=0.5)
        plt.xticks(y_pos, top_5_features)
        plt.ylabel("wT * x")
        title = "SVM Feature Analysis: Binary Classification" 
        plt.title(title)
        plt.show()
