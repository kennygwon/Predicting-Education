"""
Creates a class to preprocess the data
Authors: Kenny, Raymond, Real Rick
Date: 4/25/2019
"""
import random
import numpy as np

class Data:

    def __init__(self, filename):
        """
        Purpose - creates an instance of a data object
        Params - filename - name of file to be read in
        """
        self.filename = filename

        self.rawData = None

        #Train data
        self.XTrain = None
        self.yTrain = None
        #Validation data
        self.XVal = None
        self.yVal = None
        #Test data
        self.XTest = None
        self.yTest = None

        #SVM Data
        self.SVMTrain = None
        self.SVMTest = None
        self.SVMValid = None
        self.SVMFeatures = None

        #Tree data
        self.DTreeDataTrain = None
        self.DTreeDataTest = None

        #Naive Bayes Data
        self.NBdataTrain = None
        self.NBdataTest = None
    def readData(self, binary=False):
        """
        Purpose - reads in the data and stores in a list of lists
        Returns - nothing, but sets the self.X and ,
        """

        #initializes our list of lists containing our data
        data = []

        #opens and reads the file
        f = open(self.filename, "r")
        lines = f.readlines()

        #appends each line to our list of lists
        for line in lines:
            featureValues = line.strip().split(", ")

            if len(featureValues) > 2:   # avoid empty lines
                #remove features we don't care about
                featureValues.pop(11) # remove capital-loss
                featureValues.pop(10) # remove capital-gain
                featureValues.pop(4) # remove education-num
                education = featureValues.pop(3)
                featureValues.pop(2) # remove fnlwgt

                #only split education into two classes
                if binary:
                    if (education in ["Preschool","1st-4th","5th-6th","7th-8th"\
                    ,"9th","10th","11th","12th","HS-grad"]):
                        data.append([0] + featureValues)
                    elif(education in ["Some-college","Bachelors","Masters",\
                    "Doctorate"]):
                        data.append([1] + featureValues)

                #split education into seven classes by default
                else:
                    if (education in ["Preschool","1st-4th","5th-6th","7th-8th"]):
                        data.append([0] + featureValues)
                    elif (education in ["9th", "10th", "11th", "12th"]):
                        data.append([1] + featureValues)
                    elif(education == "HS-grad"):
                        data.append([2] + featureValues)
                    elif(education == "Some-college"):
                        data.append([3] + featureValues)
                    elif(education == "Bachelors"):
                        data.append([4] + featureValues)
                    elif(education == "Masters"):
                        data.append([5] + featureValues)
                    elif(education == "Doctorate"):
                        data.append([6] + featureValues)

        self.rawData = data
        dataSubset = self.getSubset(30000)
        self.rawData = self.splitXY(dataSubset)[0]

        #creates the features and labels for the train data
        trainDataSubset = dataSubset[:25000]
        self.XTrain, self.yTrain  = self.splitXY(trainDataSubset)

        #creates the features and labels for the Test data
        testData = dataSubset[25000:28000]
        self.XTest, self.yTest = self.splitXY(testData)

        #creates the feature and labels for the validation set
        validData = dataSubset[28000:]
        self.XVal, self.yVal = self.splitXY(validData)

    def getSubset(self, numDataPoints):
        """
        Purpose - gets a random subset of specified size
        Params - data - the original data in list of lists
                 numDataPoints - how many datapoints to return
        Return - subset - a subset of the data
        """
        #performs the shuffling in place
        random.shuffle(self.rawData)

        #returns the specified number of data points
        return self.rawData[:numDataPoints]


    def splitXY(self, subsetData):
        """
        Purpose - We want to split the labels from the features
        Params - subsetData - The unprocessed subset of data that we are working with
        Return - List X and List y of features and class labels respectively
        """
        #initializes the X and y to be returned
        X = []
        y = []

        #iterates through all data and splits into X and y
        for line in subsetData:
            y.append(line[0])
            X.append(line[1:])

        return X,y

    def createSVMDataset(self):
        """
        Purpose - This function binarizes the dataset for SVM use
        Params - none
        Returns - nothing, but sets the self.SVMdata to the binarized features
        """
        n = len(self.rawData)

        newFeatures = []
        #add each feature by category
        newFeatures.append('age')
        newFeatures.extend(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        newFeatures.extend(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
        newFeatures.extend(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
        newFeatures.extend(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
        newFeatures.extend(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        newFeatures.extend(['Female', 'Male'])
        newFeatures.append('hours-per-week')
        newFeatures.extend(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])
        newFeatures.extend(['>50K','<=50K'])
        p = len(newFeatures)
        self.SVMFeatures = newFeatures

        # assign each new feature an index in the new array
        newFeatureDict = {}
        for index in range(p):
            newFeatureDict[newFeatures[index]] = index

        # figure out indices for continuous features in the new array
        contFeatureIndexer = {
                                0 : newFeatureDict['age'],
                                1 : newFeatureDict['hours-per-week']
        }
        newData = np.zeros([n, p])

        lineCounter = 0
        for example in self.rawData:

            #print("Original example: \n", example)

            #print("Example post-pop: \n", example)

            contFeatCounter = 0
            #print("New example-------------------------------------")
            for feature in example:
                if feature.isdigit():
                    # There are two continuous features after popping unnecessary features
                    # We use the counter to index into our continuous feature indexing dictionary
                    # to determine which column index these lie in the larger feature index
                    # dictionary
                    featureIndex = contFeatureIndexer[contFeatCounter]
                    #print("Feature index in newArray is ", featureIndex)
                    contFeatCounter += 1
                    newData[lineCounter, featureIndex] = float(feature)
                elif feature == '?':
                    pass
                else:
                    # Feature not continuous - let's
                    newData[lineCounter, newFeatureDict[feature]] = 1
            lineCounter += 1
            contFeatCounter = 0

        # Now do labels
        # What are the labels? For SVM need binary classification task
        # Pre-processing complete, set SVMdata to new output

        self.SVMTrain = newData[:25000]
        self.SVMTest = newData[25000:28000]
        self.SVMValid = newData[28000:]

    def createNBDataset(self):
        """
        Purpose - This function coverts continuous features to discrete ones
                for use with Naive Bayes.
        Params - none
        Returns - nothing
        """

        workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?']
        maritalStatus = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?']
        occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?']
        relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?']
        race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?']
        sex = ['Female', 'Male', '?']
        country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']
        income = ['>50K','<=50K', '?']

        trainn = len(self.XTrain)
        trainp = len(self.XTrain[0])

        nbDatasetTrain = np.zeros([trainn,trainp])


        for rowIndex in range(trainn):
            personData = self.XTrain[rowIndex]
            nbDatasetTrain[rowIndex][0] = (float(personData[0])-15)//10
            nbDatasetTrain[rowIndex][1] = workclass.index(personData[1])
            nbDatasetTrain[rowIndex][2] = maritalStatus.index(personData[2])
            nbDatasetTrain[rowIndex][3] = occupation.index(personData[3])
            nbDatasetTrain[rowIndex][4] = relationship.index(personData[4])
            nbDatasetTrain[rowIndex][5] = race.index(personData[5])
            nbDatasetTrain[rowIndex][6] = sex.index(personData[6])
            nbDatasetTrain[rowIndex][7] = float(personData[7])//10
            nbDatasetTrain[rowIndex][8] = country.index(personData[8])
            nbDatasetTrain[rowIndex][9] = income.index(personData[9])
        self.NBdataTrain = nbDatasetTrain

        #Naive Bayes data preprocessing is similar to decision tree preprocessing
        self.DTreeDataTrain = nbDatasetTrain

        testn = len(self.XTest)
        testp = len(self.XTest[0])

        nbDatasetTest = np.zeros([testn,testp])


        for rowIndex in range(testn):
            personData = self.XTest[rowIndex]
            nbDatasetTest[rowIndex][0] = (float(personData[0])-15)//10
            nbDatasetTest[rowIndex][1] = workclass.index(personData[1])
            nbDatasetTest[rowIndex][2] = maritalStatus.index(personData[2])
            nbDatasetTest[rowIndex][3] = occupation.index(personData[3])
            nbDatasetTest[rowIndex][4] = relationship.index(personData[4])
            nbDatasetTest[rowIndex][5] = race.index(personData[5])
            nbDatasetTest[rowIndex][6] = sex.index(personData[6])
            nbDatasetTest[rowIndex][7] = float(personData[7])//10
            nbDatasetTest[rowIndex][8] = country.index(personData[8])
            nbDatasetTest[rowIndex][9] = income.index(personData[9])
        self.NBdataTest = nbDatasetTest

        #Naive Bayes data preprocessing is similar to decision tree preprocessing
        self.DTreeDataTest = nbDatasetTest

        return
