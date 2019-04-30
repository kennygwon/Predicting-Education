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
        self.X = None 
        self.y = None
        self.SVMdata = None
        self.SVMlabels = None
        self.DTreeData = None
        self.NBdata = None

    def readData(self):
        """
        Purpose - reads in the data and stores in a list of lists
        Returns - nothing, but sets the self.X and , 
        """

        #initializes our list of lists containing our data
        data = []
        rawData = []

        #opens and reads the file
        f = open(self.filename, "r")
        lines = f.readlines()

        #appends each line to our list of lists
        for line in lines:
            featureValues = line.strip().split(", ")
            if len(featureValues)  > 2:   #avoid empty lines
                rawData.append(featureValues[:-1])
            
            #gets rid of the last line which is "\n"
            if len(featureValues) > 2:
                education = featureValues.pop(3)

                if (education in ["Preschool","1st-4th","5th-6th","7th-8th"]):
                    data.append([0]+featureValues)
                elif (education in ["9th", "10th", "11th", "12th"]):
                    data.append([1]+featureValues)
                elif(education == "HS-grad"):
                    data.append([2]+featureValues)
                elif(education == "Some-college"):
                    data.append([3]+featureValues)
                elif(education == "Bachelors"):
                    data.append([4]+featureValues)
                elif(education == "Masters"):
                    data.append([5]+featureValues)
                elif(education == "Doctorate"):
                    data.append([6]+featureValues)
                
        #TODO: maybe keep track of all possible values for discrete features
        #TODO: keep track of names and indices of discrete features

        educationDictionary = {}
        for index in range(len(data)):
            try:
                educationDictionary[data[index][0]]+=1
            except:
                educationDictionary[data[index][0]]=1

        totalCount = 0
        for key in list(educationDictionary.keys()):
            totalCount += educationDictionary[key]

        self.rawData = rawData
        self.X, self.y  = self.splitXY(data)
        print(educationDictionary)

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
        p = 86 # length of new features
        newFeatures = []
        #add features by category
        newFeatures.append('age')
        newFeatures.extend(['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        newFeatures.append('education-num')
        newFeatures.extend(['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
        newFeatures.extend(['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
        newFeatures.extend(['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
        newFeatures.extend(['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
        newFeatures.extend(['Female', 'Male'])
        newFeatures.append('hours-per-week')
        newFeatures.extend(['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

        newFeatureDict = {}
        for index in range(p):
            newFeatureDict[newFeatures[index]] = index

        contFeatureIndexer = {
                                            0 : newFeatureDict['age'], 
                                            1 : newFeatureDict['education-num'], 
                                            2 : newFeatureDict['hours-per-week'] 
        }
        newData = np.zeros([n, p])        

        lineCounter = 0
        for example in self.rawData:

            #print("Original example: \n", example)
            example.pop(2) # remove fnlwgt
            example.pop(2) # remove education
            example.pop(9) # remove capital gain
            example.pop(9) # remove capital loss
            #print("Example post-pop: \n", example)

            contFeatCounter = 0
            #print("New example-------------------------------------")
            for feature in example:
                if feature.isdigit():
                    # There are three continuous features after popping excessive features above
                    # We use the counter to index into our continuous feature indexing dictionary
                    # to determine which column index these lie in the larger feature index 
                    # dictionary
                    featureIndex = contFeatureIndexer[contFeatCounter]
                    #print("Feature index in newArray is ", featureIndex)
                    contFeatCounter += 1
                    newData[lineCounter, featureIndex] = int(feature)
                elif feature == '?':
                    #TODO: What if feature is missing? Doing nothing for now
                    pass
                else:
                    # Feature not continuous - let's 
                    newData[lineCounter, newFeatureDict[feature]] = 1
            lineCounter += 1
            contFeatCounter = 0
        
        # Now do labels
        # What are the labels? For SVM need binary classification task
        # Pre-processing complete, set SVMdata to new output
        self.SVMdata = newData

    def createNBDataset(self):
        """
        Purpose - This function binarizes the dataset for SVM use
        Params - none
        Returns - nothing, but sets the self.SVMdata to the binarized features
        """
        pass

    def createDTreeDataset(self):
        """
        Purpose - This function binarizes the dataset for SVM use
        Params - none
        Returns - nothing, but sets the self.SVMdata to the binarized features
        """
        pass