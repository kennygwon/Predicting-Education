"""
File for running our algorithms on the dataset
Authors: Kenny, Raymond, Real Rick
Date: 4/25/2019
"""

from preprocess import *

def main():
    data = Data("adult.data")
    unprocessedData = data.readData()
    unproccesedSubset = data.getSubset(unprocessedData, 2000)
    X,y = data.splitXY(unprocessedSubset)
if __name__ == "__main__":
    main()
