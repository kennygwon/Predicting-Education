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
    
if __name__ == "__main__":
    main()
