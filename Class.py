"""
classifier for ML Assignment Using dataset
"""
import os
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class ProcessData:
    """A class to help with making pd.DataFrames and making useful functions on them"""
    def __init__(self, folder: str = None, filename: str = None, read: bool = True, data: List[pd.DataFrame] = None,
                 tt: bool = False):
        """An init function for the class, can be read in from file, or from passed in data"""
        if tt:  # If the incoming data already has test and training sets
            self.dataset = data[0].copy()
            self.train_set = ProcessData(read=False, data=[data[1]])
            self.test_set = ProcessData(read=False, data=[data[2]])
        elif read:  # If not, if it needs to read from a file
            self.file_path = os.path.join(folder, filename)
            self.dataset = pd.read_csv(self.file_path)
            self.train_set = None
            self.test_set = None
        else:  # Else just create one from the passed in data
            self.dataset = data[0].copy()
            self.train_set = None
            self.test_set = None

    def head(self):
        """Function to print pd.DataFrame.head()"""
        print(self.dataset.head())

    def info(self):
        """Function to print pd.DataFrame.info()"""
        print(self.dataset.info())

    def count(self, column: str):
        """Function to print the value counts on certain column"""
        print(self.dataset[column].value_counts())

    def describe(self):
        """Function to print the pd.DataFrame.describe()"""
        print(self.dataset.describe())

    def hist(self, bins: int = 100):
        """Function to show hist of the dataset"""
        self.dataset.hist(bins=bins)
        plt.show()

    def test_train_split(self, test_size: float = 0.2, random_state: int = 42):
        """Function to use sklearn.train_test_split on the data and create a test and training set which are new
        ProcessData classes so can have the same functions run on them"""
        tr_set, te_set = train_test_split(self.dataset, test_size=test_size, random_state=random_state)
        self.test_set = ProcessData(read=False, data=[te_set])
        self.train_set = ProcessData(read=False, data=[tr_set])

    def label_encode(self, column: int, new_name: str):
        """Creates a label encoder on a certain column"""
        label_encoder = LabelEncoder()
        data = self.dataset.iloc[:, column]
        stuff = label_encoder.fit_transform(np.array(data))
        self.dataset.loc[:, new_name] = stuff
        print(label_encoder.classes_)

    def corr(self):
        """Function to return a correlation matrix of the dataset"""
        return self.dataset.corr()

    def drop(self, column: str, axis: int = 1, inplace: bool = True):
        """Function to drop the specified column from the dataset, can specify if inplace or not (not by default)"""
        if inplace:
            if self.test_set:  # Checking to see if the test/train split is already in place
                self.dataset.drop(column, axis=axis, inplace=inplace)
                self.test_set.dataset.drop(column, axis=axis, inplace=inplace)
                self.train_set.dataset.drop(column, axis=axis, inplace=inplace)
            else:
                self.dataset.drop(column, axis=axis, inplace=inplace)
        else:
            if self.test_set:
                data = [self.dataset.drop(column, axis=axis), self.train_set.dataset.drop(column, axis=axis),
                        self.test_set.dataset.drop(column, axis=axis)]
                return ProcessData(read=False, data=data, tt=True)
            else:
                data = [self.dataset.drop(column, axis=axis)]
                return ProcessData(read=False, data=data)

