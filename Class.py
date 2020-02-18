import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FOLDER = './data/'


class ProcessData:
    def __init__(self, folder, filename):
        self.file_path = os.path.join(folder, filename)
        self.dataset = pd.read_csv(self.file_path)
        self.train_set = None
        self.test_set = None

    def head(self):
        print(self.dataset.head())

    def info(self):
        print(self.dataset.info())

    def count(self, column):
        print(self.dataset[column].value_counts())

    def describe(self):
        print(self.dataset.describe())

    def hist(self, bins=100):
        self.dataset.hist(bins=bins)
        plt.show()

    def test_train_split(self, test_size=0.2, random_state=42):
        self.train_set, self.test_set = train_test_split(self.dataset, test_size=test_size, random_state=random_state)

    def label_encode(self, column, new_name):
        labelencoder = LabelEncoder()
        data = self.dataset.iloc[:, column]
        stuff = labelencoder.fit_transform(np.array(data))
        self.dataset.loc[:, new_name] = stuff


studentInfo = ProcessData(FOLDER, 'studentInfo.csv')
