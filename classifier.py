from Class import ProcessData
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
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

    studentInfo.train_set.dataset.plot(kind='scatter', x='studied_creduts', y='result', alpha=0.1)
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
    big2_cat = big2.drop("studied_creduts", 1)
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
# student_assessment.info()
assessments = ProcessData(folder=FOLDER, filename="assessments.csv")
# assessments.info()

test = student_assessment.dataset.merge(assessments.dataset, how='left')
student_info = ProcessData(folder=FOLDER, filename='studentInfo.csv')
test = ProcessData(data=[test.merge(student_info.dataset, how='left')], read=False)

test = ProcessData(read=False, data=[test.dataset.dropna(axis=0)])
test.drop('date')
test.drop('date_submitted')
test.drop('is_banked')
test.label_encode(15, 'result')
length = len(test.dataset.final_result)
for i in range(length):
    if test.dataset.final_result.iat[i] == 'Withdrawn':
        test.dataset.result.iat[i] = 0
    elif test.dataset.final_result.iat[i] == 'Fail':
        test.dataset.result.iat[i] = 1
    elif test.dataset.final_result.iat[i] == 'Pass':
        test.dataset.result.iat[i] = 2
    elif test.dataset.final_result.iat[i] == 'Distinction':
        test.dataset.result.iat[i] = 3
test.drop('final_result')
test.drop('code_module')
test.drop('code_presentation')
test.drop('region')
test.label_encode(7, 'imd_band_')
test.drop('imd_band')
test.label_encode(7, 'age_band_')
test.drop('age_band')
test.label_encode(9, 'disability_')
test.drop('disability')
test.label_encode(5, 'gender_')
test.drop('gender')
test.drop('assessment_type')
test.label_encode(4, 'highest_education_')
for i in range(length):
    if test.dataset.highest_education_.iat[i] == 0:  # If it is A Level
        test.dataset.highest_education_.iat[i] = 2
    if test.dataset.highest_education_.iat[i] == 1:  # If it is HE
        test.dataset.highest_education_.iat[i] = 3
    if test.dataset.highest_education_.iat[i] == 2:  # If it is Lower than A Level
        test.dataset.highest_education_.iat[i] = 1
    if test.dataset.highest_education_.iat[i] == 3:  # If it is No Formal quals
        test.dataset.highest_education_.iat[i] = 0
print(np.array(['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification',
                'Post Graduate Qualification']))
test.drop('highest_education')
# test.info()
test.test_train_split()
test_labels = ProcessData(read=False, tt=True, data=[test.dataset['result'], test.train_set.dataset['result'],
                                                     test.test_set.dataset['result']])
test.drop('result')

lin_reg = LinearRegression()
lin_reg.fit(test.train_set.dataset, test_labels.train_set.dataset)
some_data = test.train_set.dataset.iloc[:5]
print("Predictions:", lin_reg.predict(some_data))
some_labels = test_labels.train_set.dataset.iloc[:5]
print("Labels:", list(some_labels))

test_predictions = lin_reg.predict(test.train_set.dataset)
lin_mse = mean_squared_error(test_labels.train_set.dataset, test_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(test.train_set.dataset, test_labels.train_set.dataset)
test_predictions = tree_reg.predict(test.train_set.dataset)
tree_mse = mean_squared_error(test_labels.train_set.dataset, test_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)  # OVER FITTED!!!
scores = cross_val_score(tree_reg, test.train_set.dataset, test_labels.train_set.dataset,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())


display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, test.train_set.dataset, test_labels.train_set.dataset,
                             scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
