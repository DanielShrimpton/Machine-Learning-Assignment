import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix

FOLDER = './data/'


class ProcessData:
    def __init__(self, folder=None, filename=None, read=False, data=None, tt=False):
        if tt:
            self.dataset = data[0]
            self.train_set = ProcessData(read=False, data=data[1])
            self.test_set = ProcessData(read=False, data=data[2])
        elif read:
            self.file_path = os.path.join(folder, filename)
            self.dataset = pd.read_csv(self.file_path)
            self.train_set = None
            self.test_set = None
        else:
            self.dataset = data

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
        tr_set, te_set = train_test_split(self.dataset, test_size=test_size, random_state=random_state)
        self.test_set = ProcessData(read=False, data=te_set)
        self.train_set = ProcessData(read=False, data=tr_set)

    def label_encode(self, column, new_name):
        label_encoder = LabelEncoder()
        data = self.dataset.iloc[:, column]
        stuff = label_encoder.fit_transform(np.array(data))
        self.dataset.loc[:, new_name] = stuff

    def corr(self):
        return self.dataset.corr()

    def drop(self, column, axis):
        data = [self.dataset.drop(column, axis=axis), self.train_set.dataset.drop(column, axis=axis), self.test_set.dataset.drop(column, axis=axis)]
        return ProcessData(read=False, data=data, tt=True)


studentInfo = ProcessData(folder=FOLDER, filename='studentInfo.csv', read=True)
studentInfo.label_encode(11, 'result')
studentInfo.count('final_result')
studentInfo.count('result')
studentInfo.test_train_split()
corr_matrix = studentInfo.train_set.corr()
print(corr_matrix['result'].sort_values(ascending=False))
scatter_matrix(studentInfo.train_set.dataset)
plt.show()

studentInfo.train_set.dataset.plot(kind='scatter', x='studied_credits', y='result', alpha=0.1)
plt.show()

studentInfo2 = studentInfo.drop('result', 1)
studentInfo2 = studentInfo2.drop('final_result', 1)
