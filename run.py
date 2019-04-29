"""
File for running our algorithms on the dataset
Authors: Kenny, Raymond, Real Rick
Date: 4/25/2019
"""

from preprocess import *

def main():
    data = Data("adult.data")
    unprocessedData = data.readData()
    print(unprocessedData[0])
    print(len(unprocessedData))
    subset = data.getSubset(unprocessedData, 2000)
    print(subset[0])
    print(len(subset))
if __name__ == "__main__":
    main()
