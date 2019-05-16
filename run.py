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
import optparse, sys
import matplotlib.pyplot as plt

def parse_opts(opts, parser):
    """
    Purpose - 
    Param   - opts
              parser
    Returns - 
    """
    mandatory = 'task'
    if not opts.__dict__[mandatory]:
        print('mandatory option ' + mandatory + ' is missing!\n')
        parser.print_help()
        sys.exit(1) 
    
    # now figure out which task to do
    task = opts.task
    if task == 'binary':
        return True
    elif task == 'multiclass':
        return False
    else:
        print("ERROR: Unrecognized task! Use options 'binary' or 'multiclass'\n")
        parser.print_help()
        sys.exit(1)

def main():

    parser = optparse.OptionParser(description='main.py')
    parser.add_option('-t', '--task', \
        help='whether to perform a binary/multiclass classification task or not') 

    opts = parser.parse_args()[0]

    binary = parse_opts(opts, parser)
    data = Data("adult.data")
    if binary:
        data.readData(binary=True)
    else:
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
    svm_score = svmClassifier.evaluate(data.SVMTest, data.yTest, predictions)
    #svmClassifier.visualizeWeights(data.SVMFeatures)
    # uncomment this code to perform hyperparameter tuning for the SVC 
    # classifier
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

    dTreeFeats = data.createNBDataset()

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
    nb_score = naiveBayesClassifier.evaluate(nbTestY, nbPredictions)

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
  
    print(len(treeTrainX))

    #train the decision tree model
    decisionTreeClassifier.trainTree(treeTrainX, treeTrainY)

    print("Tree is fully Grown!\n")

    #Test our model on the test data and get predictions
    predictions = decisionTreeClassifier.testTree(treeTestX)
    #evaluate the accuracy
    dtree_score = decisionTreeClassifier.evaluate(treeTestX, treeTestY, predictions)
    #decisionTreeClassifier.visualize(dTreeFeats)

    #MFC
    print("\n====================================================")
    print("Most Frequent Class Baseline")
    print("====================================================")
    mfc_score = MFC.evaluate(nbTrainY, nbTestY)

    # plot comparative bar graph
    scores = [svm_score, nb_score, dtree_score, mfc_score]
    x_labels = ['SVM', 'Naive Bayes', 'DTree', 'MFC']
    y_pos = np.arange(4)
    plt.bar(y_pos, scores,  align='center', width=0.5)
    plt.xticks(y_pos, x_labels)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Scores for Different Classifiers")
    plt.show()

if __name__ == "__main__":
    main()
