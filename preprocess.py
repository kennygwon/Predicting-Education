"""
Creates a class to preprocess the data
Authors: Kenny, Raymond, Real Rick
Date: 4/25/2019
"""
import random


class Data:

    def __init__(self, filename):
        """
        Purpose - creates an instance of a data object
        Params - filename - name of file to be read in
        """
        self.filename = filename

    def readData(self):
        """
        Purpose - reads in the data and stores in a list of lists
        Returns - a list of list of continuous and discrete values
        """

        #initializes our list of lists containing our data
        data = []

        #opens and reads the file
        f = open(self.filename, "r")
        lines = f.readlines()


        #appends each line to our list of lists
        for line in lines:
            featureValues = line.strip().split(", ")
            
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


        print(educationDictionary)
        return data

    def getSubset(self, data, numDataPoints):
        """
        Purpose - gets a random subset of specified size
        Params - data - the original data in list of lists 
                 numDataPoints - how many datapoints to return
        Return - subset - a subset of the data
        """

        #performs the shuffling in place
        random.shuffle(data)

        #returns the specified number of data points
        return data[:numDataPoints]


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
            x.append(line[1:])
        
        return X,y
