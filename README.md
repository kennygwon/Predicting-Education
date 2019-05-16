## CS 66 Final Project Lab Notebook

Name 1: Kenneth Gwon (Kenny)  
Name 2: Raymond Liu   
Name 3: Richard Muniu (Rick)  

Username 1: kgwon1  
Username 2: rliu5  
Username 3: rmuniu1  

Project Title: Census Data To Predict Levels Of Education  

---

#### Kenny: 04-25-19 (.5hrs)
* Created list of tasks
* Uploaded data file to repo
* Got started reading the data in

#### Raymond/Rick/Kenny: 04-29-19 (2hrs)
* Data preprocessing for SVM to make data suitable for use with SVM (unfinshed)
* Outlined other necessary data preprocessing that we needed to make so the data would be suitable for decision trees and naive Bayes 

#### Richard/Raymond: 04-30-19 (4hrs)
* SVM binarization of data
* Used test SVC classifier for one-vs-rest classification - extremely low test score

#### Richard/Raymond: 5-2-19 (2 hrs)
* Used K-fold cross-validation and hyperparameter tuning for SVM to make accuracy scores better. No improvement
* Worked on the framework for the decision tree model (More data preprocessing)

#### Richard/Raymond: 5-5-19 (2hrs)
* SVM optimization with C in np.logspace(-10, 10, 21) - better accuracy. Optimal C = 0.1.
* Decision Tree implementation and calls (main.py) and confusion matrices.

#### Kenny: 5-7-19 (1.5hrs)
* Preprocessing refactoring to support binary classification task - collapsed labels into college vs. no college
* Implemented and applied Naive Bayes

#### Richard/Raymond/Kenny: 5-12-19 (2.5hrs)
* Command line argument parsing and code refactoring (main.py) to allow for binary/multiclass classification options.
* Decision Tree visualization using graphviz and various tree depths
* Noticed ordinal feature weight problem with Naive Bayes dataset - changed format to one hot encoding, support a class for unknown types (‘?’)
* Presentation slides initialization

#### Richard/Raymond/Kenny: 5-13-19 (.5hrs)
* Final presentation prep huddle and to-do list creation.
* Decision Tree feature analysis 
* SVM feature analysis attempt (multiclass classification task)

#### Richard/Raymond/Kenny: 5-15-19 (3hrs)
* Presentation slides finalization and clean-up
* Evaluation chart generation
* SVM binary feature analysis.

---

## References
Census Income Data Set , https://archive.ics.uci.edu/ml/datasets/Census+Income, Ronny Kohavi and Barry Becker (1996)

Lemon, Zelazo, Mulakaluri, “Predicting if income exceeds $50,000 per year based on 1994 US Census Data with Simple Classification Techniques”, http://cseweb.ucsd.edu/classes/sp15/cse190-c/reports/sp15/048.pdf, 

Sklearn naive_bayes.MultinomialNB, retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html on May 7, 2019

Sklearn svm.LinearSVC, retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html on April 29, 2018

Sklearn tree.DecisionTreeClassifier, retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html on May 5, 2019


