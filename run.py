"""
File for running our algorithms on the dataset
Authors: Kenny, Raymond, Real Rick
Date: 4/25/2019
"""

from preprocess import *
from sklearn import svm  
from sklearn.metrics import confusion_matrix, accuracy_score 
from SVM import *
from naiveBayes import *
from decisionTrees import *

def main():
    data = Data("adult.data")
    data.readData()
    data.createSVMDataset()
    
    print("\n====================================================")
    print("SVM Revving Up...")
    print("====================================================")
    #creates multi-class SVM model
    svmClassifier = SVM()
    #train SVM model
    svmClassifier.trainSVM(data.SVMTrain, data.yTrain)
    #evaluate test data
    predictions = svmClassifier.testSVM(data.SVMTest)
    # evaluate accuracy and print confusoin matrix
    svmClassifier.evaluate(data.SVMTest, data.yTest, predictions)
    """
    print("\n====================================================")
    print("\t Starting hyperparameter tuning...")
    print("====================================================")
    svc_params = {"C": np.logspace(0, 3, 4), "gamma": np.logspace(-4, 0, 5)}
    svmClassifier.runTuneTest(svc_params, data.SVMTrain, data.yTrain)
    svmClassifier.printTestScores()
    """

    #DecisionTree 
    data.createNBDataset()
    
    print("\n====================================================")
    print("Planting Decision Tree Seeds...")
    print("====================================================")
    print("Watering soil...")
    print("====================================================")

    #creates decision tree
    decisionTreeClassifier = DecisionTree()
    #split the train and test data into example and labels
    treeTrainX, treeTrainY = data.splitXY(data.DTreeDataTrain)
    treeTestX, treeTestY = data.splitXY(data.DTreeDataTest)
    #train the decision tree model
    decisionTreeClassifier.trainTree(treeTrainX, treeTrainY)

    print("Tree is fully Grown!\n")

    #Test our model on the test data and get predictions
    predictions = decisionTreeClassifier.testTree(treeTestX)
    #evaluate the accuracy
    decisionTreeClassifier.evaluate(treeTestX, treeTestY, predictions)


if __name__ == "__main__":
    main()
