"""
Creates a class to preprocess the data
Authors: Kenny, Raymond, Real Rick
Date: 4/25/2019
"""

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
            data.append(line.split(","))

        #TODO: maybe keep track of all possible values for discrete features
        #TODO: keep track of names and indices of discrete features

        return data


