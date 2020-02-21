"""
classifier for ML Assignment Using dataset
"""
import os
from typing import List
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from mpl_toolkits.mplot3d import Axes3D

FOLDER = './data/'  # folder that contains the csv data files


class ProcessData:
    """A class to help with making pd.DataFrames and making useful functions on them"""
    def __init__(self, folder: str = None, filename: str = None, read: bool = False, data: List[pd.DataFrame] = None,
                 tt: bool = False):
        """An init function for the class, can be read in from file, or from passed in data"""
        if tt:  # If the incoming data already has test and training sets
            self.dataset = data[0]
            self.train_set = ProcessData(read=False, data=[data[1]])
            self.test_set = ProcessData(read=False, data=[data[2]])
        elif read:  # If not, if it needs to read from a file
            self.file_path = os.path.join(folder, filename)
            self.dataset = pd.read_csv(self.file_path)
            self.train_set = None
            self.test_set = None
        else:  # Else just create one from the passed in data
            self.dataset = data[0]
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

    def drop(self, column: str, axis: int, inplace: bool = False):
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

# studentInfo2 = studentInfo.drop('result', 1)
# studentInfo2 = studentInfo.drop('final_result', 1)

studentAssessment = ProcessData(folder=FOLDER, filename='studentAssessment.csv', read=True)
studentAssessment.info()
studentAssessment.describe()
studentAssessment.hist()
plt.show()
# scatter_matrix(studentAssessment.dataset)
# plt.show()
studentAssessment.dataset.plot(kind='scatter', x='id_assessment', y='score', alpha=0.1, c='id_student',
                               cmap=plt.get_cmap("jet"), colorbar=True)
plt.show()
studentAssessment.dataset.plot(kind='scatter', x='id_student', y='score', alpha=0.1, c='id_assessment',
                               cmap=plt.get_cmap('jet'), colorbar=True)
plt.show()

studentAssessment.test_train_split()

z = studentAssessment.test_set.dataset.score
y = studentAssessment.test_set.dataset.id_student
x = studentAssessment.test_set.dataset.id_assessment

fig = plt.figure()
ax = plt.axes(projection='3d')
color_fig = ax.scatter(x, y, z, cmap=plt.get_cmap('jet'), c=y, alpha=0.1)
ax.set_xlabel('id_assessment')
ax.set_ylabel('id_student')
ax.set_zlabel('score')
fig.colorbar(color_fig)
plt.show()

big = ProcessData(read=False, data=[pd.concat([studentAssessment.dataset, studentInfo.dataset], sort=True)])
big.count('id_student')
big2 = big.drop("age_band", 1)
big2.drop("code_module", 1, inplace=True)
big2.drop("code_presentation", 1, inplace=True)
big2.drop("date_submitted", 1, inplace=True)
big2.drop("highest_education", 1, inplace=True)
big2.drop("imd_band", 1, inplace=True)
big2.drop("is_banked", 1, inplace=True)
big2.drop("region", 1, inplace=True)
# big2.drop("final_result", 1, inplace=True)
big2.info()
big2_cat = big2.drop("studied_credits", 1)
big2_cat.drop("score", 1, inplace=True)
big2_cat.drop("result", 1, inplace=True)
big2_cat.drop("num_of_prev_attempts", 1, inplace=True)
big2_cat.drop("id_student", 1, inplace=True)
big2_cat.drop("id_assessment", 1, inplace=True)
big2_cat = ProcessData(read=False, data=[big2_cat.dataset.dropna()])

cat_encoder = OneHotEncoder()
big3 = cat_encoder.fit_transform(big2_cat.dataset)
print(cat_encoder.categories_)
print(big3)
print(big3.toarray())

num = 537811
length = len(big2.dataset.id_student)
results = [big2.dataset.final_result.iat[x] for x in range(length) if big.dataset.id_student.iat[x] == num]
results_mini = [studentInfo.dataset.final_result.iat[x] for x in range(len(studentInfo.dataset.id_student)) if
                studentInfo.dataset.id_student.iat[x] == num]
print(results, results_mini)
