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

def main():
    data = Data("adult.data")
    data.readData()
    data.createSVMDataset()
    
    print("\n====================================================")
    print("STARTING SVM...")
    print("====================================================")
    #creates multi-class SVM model
    svmClassifier = SVM()
    #train SVM model
    svmClassifier.trainSVM(data.SVMTrain, data.yTrain)
    #evaluate test data
    predictions = svmClassifier.testSVM(data.SVMTest)
    # evaluate accuracy and print confusoin matrix
    svmClassifier.evaluate(data.SVMTest, data.yTest, predictions)
    print("\n====================================================")
    print("\t Starting hyperparameter tuning...")
    print("====================================================")
    svc_params = {"C": np.logspace(0, 3, 4), "gamma": np.logspace(-4, 0, 5)}
    svmClassifier.runTuneTest(svc_params, data.SVMTrain, data.yTrain)
    svmClassifier.printTestScores()
    
if __name__ == "__main__":
    main()
