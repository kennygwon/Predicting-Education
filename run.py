"""
File for running our algorithms on the dataset
Authors: Kenny, Raymond, Real Rick
Date: 4/25/2019
"""

from preprocess import *
from sklearn import svm  



def main():
    data = Data("adult.data")
    data.readData()
    data.createSVMDataset()
    
    #creates multi-class SVM model
    svmClassifier = svm.LinearSVC()
    svmClassifier.fit(data.SVMTrain, data.yTrain)
    svmPredictions = svmClassifier.predict(data.SVMTrain)
    svmScore = svmClassifier.score(data.SVMTrain, data.yTrain)
    print("SVM classifer score was ", svmScore)
    
if __name__ == "__main__":
    main()
