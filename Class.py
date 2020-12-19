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
    """A class to help with making pd.DataFrames and making useful functions on them
    My Custom class to handle pandas datasets with custom, repeated functions"""

    def __init__(self, folder: str = None, filename: str = '.', read: bool = True,
                 data: List[pd.DataFrame] = None, tt: bool = False):
        """
        An init function for the class, can be read in from file, or from passed in data

        :param folder: The folder the file is located in if reading in a file (default current
        working directory)

        :param filename: The filename of the data to read in

        :param read: Boolean to say whether it needs to read from a file or not

        :param data: A list containing the datasets of either length 1 with no test/train or
        length 3 with [dataset, train, test]

        :param tt: Boolean to say whether incoming data already has test and training sets
        """
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
        """Function to use sklearn.train_test_split on the data and create a test and training
        set which are new ProcessData classes so can have the same functions run on them"""
        tr_set, te_set = train_test_split(self.dataset, test_size=test_size,
                                          random_state=random_state)
        self.test_set = ProcessData(read=False, data=[te_set])
        self.train_set = ProcessData(read=False, data=[tr_set])

    def label_encode(self, name: str, new_name: str, drop: bool = True):
        """Creates a label encoder on a certain column"""
        column = self.dataset.columns.get_loc(name)
        label_encoder = LabelEncoder()
        data = self.dataset.iloc[:, column]
        stuff = label_encoder.fit_transform(np.array(data))
        self.dataset.loc[:, new_name] = stuff
        print(label_encoder.classes_)
        if drop:
            self.drop(name)

    def corr(self):
        """Function to return a correlation matrix of the dataset"""
        return self.dataset.corr()

    def drop(self, column: str, axis: int = 1, inplace: bool = True):
        """Function to drop the specified column from the dataset, can specify if inplace or not
        (not by default)"""
        if inplace:
            if self.test_set:  # Checking to see if the test/train split is already in place
                self.dataset.drop(column, axis=axis, inplace=inplace)
                self.test_set.dataset.drop(column, axis=axis, inplace=inplace)
                self.train_set.dataset.drop(column, axis=axis, inplace=inplace)
            else:
                self.dataset.drop(column, axis=axis, inplace=inplace)
        else:
            if self.test_set:
                data = [self.dataset.drop(column, axis=axis),
                        self.train_set.dataset.drop(column, axis=axis),
                        self.test_set.dataset.drop(column, axis=axis)]
                return ProcessData(read=False, data=data, tt=True)
            else:
                data = [self.dataset.drop(column, axis=axis)]
                return ProcessData(read=False, data=data)


def data_one_hot(folder):
    # TODO check is one hot and if not rename to make sense
    #  Import the datasets
    student_assessment = ProcessData(folder=folder, filename="studentAssessment.csv")
    assessments = ProcessData(folder=folder, filename="assessments.csv")
    student_info = ProcessData(folder=folder, filename="studentInfo.csv")

    #  Merge student assessment and assessment
    data_ = ProcessData(data=[student_assessment.dataset.merge(assessments.dataset, how='left')],
                        read=False)
    #  Merge now with student info
    data_ = ProcessData(data=[data_.dataset.merge(student_info.dataset, how='left')], read=False)
    #  Duplicate the code_presentation field
    data_.dataset['presentation'] = data_.dataset['code_presentation']

    # Set the result column to be the final result numerically in ascending order of importance
    data_.dataset['result'] = data_.dataset['final_result'].astype('category')
    data_.dataset['result'] = data_.dataset['result']\
        .cat.reorder_categories(['Withdrawn', 'Fail', 'Pass', 'Distinction'], ordered=True)
    data_.dataset['result'] = data_.dataset['result'].cat.codes

    #  One-hot-encode all the object data
    encoded = pd.get_dummies(data_.dataset, columns=['age_band', 'imd_band', 'highest_education',
                                                     'gender', 'region', 'assessment_type',
                                                     'code_module', 'code_presentation',
                                                     'id_assessment'])
    data_ = ProcessData(data=[encoded], read=False)
    data_.label_encode('disability', 'disability_')
    # Remove the withdrawn students and drop any rows with NaN in
    data_ = ProcessData(data=[data_.dataset.query('final_result != "Withdrawn"').dropna()],
                        read=False)
    data_.info()

    # Group the data
    grouped = data_.dataset.groupby(['id_student', 'presentation', 'final_result'])
    compact = grouped.first()
    # Create the sum and mean columns
    compact['score'] = grouped['score'].sum()
    compact['score_mean'] = grouped['score'].mean()
    compact['weight'] = grouped['weight'].sum()

    # Make the data reduced where any one-hot encoded columns are all zeros are dropped
    data_ = ProcessData(data=[compact.loc[:, (compact != 0).any(axis=0)]], read=False)
    # Print the correlation matrix
    corr = data_.corr()
    corr = corr.reindex(corr['result'].sort_values()
                        .reset_index(level=0).iloc[:, 0], axis=1).sort_values(by=['result'],
                                                                              ascending=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(corr["result"])

    # Split the data into test and train
    data_.test_train_split()

    # Remove the labels and make own dataset
    _data_labels = ProcessData(read=False, tt=True, data=[data_.dataset['result'],
                                                          data_.train_set.dataset['result'],
                                                          data_.test_set.dataset['result']])
    data_.drop('result')
    return data_, _data_labels
