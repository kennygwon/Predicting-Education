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

        print(len(lines))

        #appends each line to our list of lists
        for line in lines:
            featureValues = line.strip().split(",")
            #does not add the last line which is just "\n"
            if len(featureValues) > 4:
                data.append(featureValues)

        #TODO: maybe keep track of all possible values for discrete features
        #TODO: keep track of names and indices of discrete features

        educationDictionary = {}
        for index in range(len(data)):
            try:
                educationDictionary[data[index][3]]+=1
            except:
                educationDictionary[data[index][3]]=1

        totalCount = 0
        for key in list(educationDictionary.keys()):
            totalCount += educationDictionary[key]


        return data

    def getSubset(self, data, numDataPoints):
        """
        Purpose - gets a random subset of specified size
        Params - data - the original data in list of lists 
                 numDataPoints - how many datapoints to return
        Return - subset - a subset of the data
        """

        random.shuffle(data)

        return data[:numDataPoints]

