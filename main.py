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
import MFC as MFC
import optparse

def parse_opts(opts, parser):
    """
    Purpose - Parses command line arguments to decide whether or not to 
              perform binary or multiclass classification
    Params  - opts: command line arguments
              parser: parser object
    Returns - boolean, true if user wants a binary classification task, false 
              otherwise 
    """
    mandatory = 'binary'
    if not opts.__dict__[mandatory]:
        print('mandatory option ' + mandatory + ' is missing\n')
        parser.print_help()
        sys.exit(1)

def main():
    data = Data("adult.data")
    parser = optparse.OptionParser(description='main.py')
    parser.add_option('-b', '--binary', type='string', help='whether to run binary classification task')
    opts = parser.parse_args()[0]

    data.readData()
    
    # Uncomment line below to binary classification task, otherwise performs 
    # multiclass classification using 7 classes
    # data.readData(binary=True)
    
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
    print("data dimension: ", data.SVMTrain.shape)
    svmClassifier.visualizeWeights(data.SVMFeatures)

    """
    print("\n====================================================")
    print("\t Starting hyper-parameter tuning...")
    print("====================================================")
    svc_params = {"C": np.logspace(-10, 10, 21)}
    svmClassifier.runTuneTest(svc_params, data.SVMTrain, data.yTrain)
    svmClassifier.printTestScores()
    print("SVM Training Complete!")
    print("====================================================")
    """

    data.createNBDataset()

    #Naive Bayes
    print("\n====================================================")
    print("Starting Naive Bayes...")
    print("====================================================")
    print("Making naive assumptions...")
    print("====================================================")
    #splits the train and test data into example and labels
    nbTrainX = data.NBdataTrain
    nbTrainY = data.yTrain
    nbTestX = data.NBdataTest
    nbTestY = data.yTest

    #trains the naive bayes classifier
    naiveBayesClassifier = NaiveBayes()
    naiveBayesClassifier.trainNB(nbTrainX, nbTrainY)

    #test our model on the test data and get predictions
    nbPredictions = naiveBayesClassifier.testNB(nbTestX)
    #evaluate the accuracy of nb model
    naiveBayesClassifier.evaluate(nbTestY, nbPredictions)



    #DecisionTree
    print("\n====================================================")
    print("Planting Decision Tree Seeds...")
    print("====================================================")
    print("Watering soil...")
    print("====================================================")

    #creates decision tree
    decisionTreeClassifier = DecisionTree()
    #split the train and test data into example and labels
    treeTrainX = data.DTreeDataTrain
    treeTrainY = data.yTrain
    treeTestX = data.DTreeDataTest
    treeTestY = data.yTest

    #train the decision tree model
    decisionTreeClassifier.trainTree(treeTrainX, treeTrainY)

    print("Tree is fully Grown!\n")

    #Test our model on the test data and get predictions
    predictions = decisionTreeClassifier.testTree(treeTestX)
    #evaluate the accuracy
    decisionTreeClassifier.evaluate(treeTestX, treeTestY, predictions)


    #MFC
    print("\n====================================================")
    print("Most Frequent Class Baseline")
    print("====================================================")
    MFC.evaluate(nbTrainY, nbTestY)


if __name__ == "__main__":
    main()
