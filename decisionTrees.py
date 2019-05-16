"""
Uses decision trees to train a model
Authors: Kenny Gwon, Raymond Liu, Richard Muniu
date: 5/8/19
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

class DecisionTree:

  def __init__(self):
    #Makes the decision tree classifier a member variable
    self.dTree = DecisionTreeClassifier(random_state=42)

  def trainTree(self, trainX, trainy):
    """
    Purpose: Trains a decision tree using our training data
    Params: trainX - Training examples
                trainy - Training labels
    Return: returns a decision tree classifier
    """

    #Builds a decision tree classifier from the training set
    self.dTree.fit(trainX, trainy)

  def testTree(self, testX):
    """
    Purpose: Uses test data on the decision tree model we created
    Params: testX: Test data / test examples
    Return: a list of predicted y values
    """

    #The predict method will return a list of predicted y values from the test examples
    yPred = self.dTree.predict(testX)

    return yPred

  def evaluate(self, testX, testy, yPred):
    """
    Purpose: Evaluates our tree model
    Params: testy: The true labels / target labels
    """

    #The score method will return the mean accuracy on the given test data
    #and labels
    treeScores = self.dTree.score(testX, testy)
    print("The Decision Tree's Score is...", treeScores)

    print("Confusion Matrix")
    confusionMatrix = confusion_matrix(testy, yPred)
    print(confusionMatrix)
    print()

    return treeScores

  def visualize(self, dTreeFeats):
    """
    Purpose: Prints out the decision tree so we can visualize it
    Params: dTreeFeats - a list of feature names
    Return: String representation of the decision tree
    """
    dot_data = StringIO()

    export_graphviz(self.dTree, out_file=dot_data,
                    max_depth=3, feature_names= dTreeFeats,
                    filled=True, rounded=True,
                    rotate=True,
                    special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    graph.write_png("tree.png")
