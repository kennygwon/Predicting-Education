"""
File for running our algorithms on the dataset
Authors: Kenny, Raymond, Real Rick
Date: 4/25/2019
"""

from preprocess import *

def main():
    data = Data("adult.data")
    data.readData()
    unprocessedSubset = data.getSubset(2000)
    data.createSVMDataset()
    print(data.SVMdata.shape)

if __name__ == "__main__":
    main()
