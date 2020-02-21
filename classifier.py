from Class import ProcessData
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from mpl_toolkits.mplot3d import Axes3D

FOLDER = './data/'  # folder that contains the csv data files


def first_attempt():
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


student_assessment = ProcessData(folder=FOLDER, filename="studentAssessment.csv")
student_assessment.info()
assessments = ProcessData(folder=FOLDER, filename="assessments.csv")
assessments.info()
