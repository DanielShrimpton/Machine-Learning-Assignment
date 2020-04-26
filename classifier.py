import os
import datetime
import sys
import time
from Class import ProcessData
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, \
    StratifiedKFold, validation_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, \
    classification_report, max_error, balanced_accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
import numbers

start = time.time()

FOLDER = './data/'  # folder that contains the csv data files


def first_attempt():
    student_info = ProcessData(folder=FOLDER, filename='studentInfo.csv', read=True)
    student_info.label_encode('final_result', 'result')
    student_info.count('final_result')
    student_info.count('result')
    student_info.test_train_split()
    corr_matrix = student_info.train_set.corr()
    print(corr_matrix['result'].sort_values(ascending=False))
    scatter_matrix(student_info.train_set.dataset)
    plt.show()

    student_info.train_set.dataset.plot(kind='scatter', x='studied_creduts', y='result', alpha=0.1)
    plt.show()

    # student_info2 = student_info.drop('result', 1)
    # student_info2 = student_info.drop('final_result', 1)

    studentAssessment = ProcessData(folder=FOLDER, filename='studentAssessment.csv', read=True)
    studentAssessment.info()
    studentAssessment.describe()
    studentAssessment.hist()
    plt.show()
    # scatter_matrix(studentAssessment.dataset)
    # plt.show()
    studentAssessment.dataset.plot(kind='scatter', x='id_assessment', y='score', alpha=0.1,
                                   c='id_student', cmap=plt.get_cmap("jet"), colorbar=True)
    plt.show()
    studentAssessment.dataset.plot(kind='scatter', x='id_student', y='score', alpha=0.1,
                                   c='id_assessment', cmap=plt.get_cmap('jet'), colorbar=True)
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

    big = ProcessData(read=False, data=[pd.concat([studentAssessment.dataset,
                                                   student_info.dataset], sort=True)])
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
    results = [big2.dataset.final_result.iat[x] for x in range(length) if
               big.dataset.id_student.iat[x] == num]
    results_mini = [student_info.dataset.final_result.iat[x] for x in
                    range(len(student_info.dataset.id_student)) if
                    student_info.dataset.id_student.iat[x] == num]
    print(results, results_mini)


def process_data():
    student_assessment = ProcessData(folder=FOLDER, filename="studentAssessment.csv")
    # student_assessment.info()
    assessments = ProcessData(folder=FOLDER, filename="assessments.csv")
    # assessments.info()

    _data = student_assessment.dataset.merge(assessments.dataset, how='left')
    student_info = ProcessData(folder=FOLDER, filename='studentInfo.csv')
    _data = ProcessData(data=[_data.merge(student_info.dataset, how='left')], read=False)

    _data = ProcessData(read=False, data=[_data.dataset.dropna(axis=0)])
    _data.drop('date')
    _data.drop('date_submitted')
    _data.drop('is_banked')
    _data.label_encode('final_result', 'result')
    # length = len(_data.dataset.final_result)
    # for i in range(length):
    #     if data.dataset.final_result.iat[i] == 'Withdrawn':
    #         data.dataset.result.iat[i] = 0
    #     elif data.dataset.final_result.iat[i] == 'Fail':
    #         data.dataset.result.iat[i] = 1
    #     elif data.dataset.final_result.iat[i] == 'Pass':
    #         data.dataset.result.iat[i] = 2
    #     elif data.dataset.final_result.iat[i] == 'Distinction':
    #         data.dataset.result.iat[i] = 3
    # _data.drop('final_result')
    _data.drop('code_module')
    _data.drop('code_presentation')
    _data.label_encode('region', 'region_')
    _data.label_encode('imd_band', 'imd_band_')
    _data.label_encode('age_band', 'age_band_')
    _data.label_encode('disability', 'disability_')
    _data.label_encode('gender', 'gender_')
    _data.drop('assessment_type')
    _data.label_encode('highest_education', 'highest_education_')
    # for i in range(length):
    #     if data.dataset.highest_education_.iat[i] == 0:  # If it is A Level
    #         data.dataset.highest_education_.iat[i] = 2
    #     if data.dataset.highest_education_.iat[i] == 1:  # If it is HE
    #         data.dataset.highest_education_.iat[i] = 3
    #     if data.dataset.highest_education_.iat[i] == 2:  # If it is Lower than A Level
    #         data.dataset.highest_education_.iat[i] = 1
    #     if data.dataset.highest_education_.iat[i] == 3:  # If it is No Formal quals
    #         data.dataset.highest_education_.iat[i] = 0
    # print(np.array(['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent',
    #                 'HE Qualification', 'Post Graduate Qualification']))
    # data.drop('highest_education')
    _data.drop('id_student')
    # _data.drop('id_assessment')
    # _data.info()
    _data.test_train_split()
    corr_mat = _data.dataset.corr()
    print(corr_mat['result'].sort_values(ascending=False))

    _data_labels = ProcessData(read=False, tt=True, data=[_data.dataset['result'],
                                                          _data.train_set.dataset['result'],
                                                          _data.test_set.dataset['result']])
    _data.drop('result')
    return _data, _data_labels


def process_data2():
    student_assessment = ProcessData(folder=FOLDER, filename="studentAssessment.csv")
    assessments = ProcessData(folder=FOLDER, filename="assessments.csv")
    student_info = ProcessData(folder=FOLDER, filename="studentInfo.csv")

    data_ = ProcessData(data=[student_assessment.dataset.merge(assessments.dataset, how='left')],
                        read=False)

    data_ = ProcessData(data=[data_.dataset.merge(student_info.dataset, how='left').dropna(axis=0)],
                        read=False)

    data_.label_encode('code_module', 'code_module_', drop=False)
    data_.label_encode('code_presentation', 'code_presentation_', drop=False)
    data_.label_encode('assessment_type', 'assessment_type_')
    data_.label_encode('gender', 'gender_')
    data_.label_encode('region', 'region_')
    data_.label_encode('highest_education', 'education')
    data_.label_encode('imd_band', 'imd_band_')
    data_.label_encode('disability', 'disability_')
    data_.label_encode('age_band', 'age_band_')
    data_.label_encode('final_result', 'result', drop=False)

    data_ = ProcessData(data=[data_.dataset.query('final_result != "Withdrawn"').dropna(axis=0)],
                        read=False)

    print('id_student no dupes: ', len(data_.dataset[['id_student']].drop_duplicates()))
    print('id_students to final result: ', len(data_.dataset[['id_student',
                                                              'final_result']].drop_duplicates()))
    print('id_student to assessment: ', len(data_.dataset[['id_student',
                                                           'id_assessment']].drop_duplicates()))
    print('id_student to code module: ', len(data_.dataset[['id_student',
                                                            'code_module']].drop_duplicates()))
    print('id_student to code_module to final_result: ',
          len(data_.dataset[['id_student', 'code_module', 'final_result']].drop_duplicates()))

    corr_mat = data_.corr()
    print(corr_mat['result'].sort_values(ascending=False))
    data_.dataset['id'] = data_.dataset['id_student'].astype(str) + data_.dataset['code_module']
    data_.count('id')
    print('id to id_assessment: ', len(data_.dataset[['id', 'id_assessment']].drop_duplicates()))
    print('id_assessment to code module: ', len(data_.dataset[['id_assessment',
                                                               'code_module']].drop_duplicates()))
    print('dataset: ', len(data_.dataset))
    print('id to final result: ', len(data_.dataset[['id', 'final_result']].drop_duplicates()))
    data_.dataset['id_result'] = data_.dataset['id'] + data_.dataset['final_result']
    data_.count('id_result')
    print(len(data_.dataset[['id_student', 'code_presentation']].drop_duplicates()))
    print(data_.dataset.query('id_result == "570213FFFFail"')
          [['id_result', 'code_presentation']].drop_duplicates())
    # data_.info()
    # corr_mat = data_.corr()
    # print(corr_mat['result'].sort_values(ascending=False))


def data3():
    student_assessment = ProcessData(folder=FOLDER, filename="studentAssessment.csv")
    assessments = ProcessData(folder=FOLDER, filename="assessments.csv")
    student_info = ProcessData(folder=FOLDER, filename="studentInfo.csv")

    student_info.info()
    student_assessment.info()
    assessments.info()

    print(len(student_info.dataset[['id_student', 'code_presentation',
                                    'final_result']].drop_duplicates()),
          len(student_info.dataset[['id_student', 'code_presentation', 'final_result']]))

    # Unique entries are id_student + code_presentation + final_result
    data_ = ProcessData(data=[student_info.dataset.query('final_result != "Withdrawn"')],
                        read=False)
    data_.dataset['id'] = data_.dataset['id_student'].astype(str) + data_.dataset[
        'code_presentation'] + data_.dataset['final_result']
    data_.count('id')

    data2 = ProcessData(data=[student_assessment.dataset.merge(assessments.dataset, how='left')],
                        read=False)

    data2 = ProcessData(data=[data2.dataset.merge(student_info.dataset, how='left')], read=False)
    data2 = ProcessData(data=[data2.dataset.query('final_result != "Withdrawn"').dropna()],
                        read=False)

    print(len(data2.dataset[['id_student', 'code_presentation', 'final_result']].drop_duplicates()))
    print(len(data2.dataset[['id_student', 'code_presentation', 'final_result',
                             'score']].drop_duplicates()),
          len(data2.dataset[['id_student', 'code_presentation', 'final_result', 'score']]))

    # data2.label_encode('final_result', 'result', drop=False)
    data2.dataset['result'] = data2.dataset['final_result']
    data2.label_encode('code_presentation', 'code_presentation_', drop=False)
    data2.label_encode('code_module', 'code_module_')
    data2.label_encode('assessment_type', 'assess_type')
    data2.label_encode('gender', 'gender_')
    data2.label_encode('region', 'region_')
    data2.label_encode('highest_education', 'edu')
    data2.label_encode('imd_band', 'imd_band_')
    data2.label_encode('age_band', 'age_band_')
    data2.label_encode('disability', 'disability_')

    data2.info()
    data2.count('final_result')

    grouped = data_.dataset.groupby(['id_student', 'code_presentation', 'final_result'])
    print(grouped.first())

    grouped2 = data2.dataset.groupby(['id_student', 'code_presentation', 'final_result'])
    print(grouped2.first())

    print(grouped2.count())
    print(grouped2['score'].sum())
    print(grouped2.get_group((6516, '2014J', 'Pass'))['id_assessment'].value_counts())
    print(grouped2.get_group((6516, '2014J', 'Pass'))['studied_credits'])
    print(grouped2.get_group((2698251, '2014B', 'Fail'))[['code_module_', 'id_assessment',
                                                          'weight']])
    ting = grouped2.first()
    ting['score'] = grouped2['score'].sum()
    ting['weight'] = grouped2['weight'].sum()
    print(ting['score'])
    print(ting['weight'])
    print(ting['num_of_prev_attempts'])
    # sys.exit()

    thingy = ProcessData(data=[ting], read=False)
    thingy.info()
    # corr_mat = thingy.corr()
    # print(corr_mat['result'].sort_values(ascending=False))

    thingy.test_train_split()

    _data_labels = ProcessData(read=False, tt=True, data=[thingy.dataset['result'],
                                                          thingy.train_set.dataset['result'],
                                                          thingy.test_set.dataset['result']])
    thingy.drop('result')
    return thingy, _data_labels


def data_one_hot():
    student_assessment = ProcessData(folder=FOLDER, filename="studentAssessment.csv")
    assessments = ProcessData(folder=FOLDER, filename="assessments.csv")
    student_info = ProcessData(folder=FOLDER, filename="studentInfo.csv")

    data_ = ProcessData(data=[student_assessment.dataset.merge(assessments.dataset, how='left')],
                        read=False)
    data_ = ProcessData(data=[data_.dataset.merge(student_info.dataset, how='left')], read=False)
    data_.info()
    # sys.exit()
    data_.dataset['presentation'] = data_.dataset['code_presentation']
    data_.dataset['result'] = data_.dataset['final_result']
    encoded = pd.get_dummies(data_.dataset, columns=['age_band', 'imd_band', 'highest_education',
                                                     'gender', 'region', 'assessment_type',
                                                     'code_module', 'code_presentation',
                                                     'id_assessment'])
    data_ = ProcessData(data=[encoded], read=False)
    data_.label_encode('disability', 'disability_')
    data_ = ProcessData(data=[data_.dataset.query('final_result != "Withdrawn"').dropna()],
                        read=False)
    data_.info()

    data_.dataset['result'] = data_.dataset['final_result']
    grouped = data_.dataset.groupby(['id_student', 'presentation', 'final_result'])
    compact = grouped.first()
    compact['score'] = grouped['score'].sum()
    compact['weight'] = grouped['weight'].sum()

    data_ = ProcessData(data=[compact], read=False)
    data_.info()

    data_.test_train_split()

    _data_labels = ProcessData(read=False, tt=True, data=[data_.dataset['result'],
                                                          data_.train_set.dataset['result'],
                                                          data_.test_set.dataset['result']])
    data_.drop('result')
    return data_, _data_labels


data, data_labels = data_one_hot()
# data, data_labels = data3()
# sys.exit()
"""process_data2(); sys.exit()
data, data_labels = process_data()"""
end = time.time()
elapsed = datetime.timedelta(seconds=(end - start))
print("Data Processing Done! Took %s" % elapsed)


# print("--- Linear Regression ---")
# lin_reg = LinearRegression()
# lin_reg.fit(data.train_set.dataset, data_labels.train_set.dataset)
# some_data = data.train_set.dataset.iloc[:5]
# print("Predictions:", lin_reg.predict(some_data))
# some_labels = data_labels.train_set.dataset.iloc[:5]
# print("Labels:", list(some_labels))
#
# test_predictions = lin_reg.predict(data.train_set.dataset)
# lin_mse = mean_squared_error(data_labels.train_set.dataset, test_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)
#
# print("--- Decision Tree Regression ---")
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(data.train_set.dataset, data_labels.train_set.dataset)
# test_predictions = tree_reg.predict(data.train_set.dataset)
# tree_mse = mean_squared_error(data_labels.train_set.dataset, test_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)  # OVER FITTED!!!
# scores = cross_val_score(tree_reg, data.train_set.dataset, data_labels.train_set.dataset,
#                          scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())


# print("--- Decision Tree Regression RMSE Scores ---")
# display_scores(tree_rmse_scores)
#
# lin_scores = cross_val_score(lin_reg, data.train_set.dataset, data_labels.train_set.dataset,
#                              scoring='neg_mean_squared_error', cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
#
# print("--- Linear Regression RMSE scores ---")
# display_scores(lin_rmse_scores)
#
# print("--- Random Forest Regressor ---")
# forest_reg1 = RandomForestRegressor()
# forest_reg1.fit(data.train_set.dataset, data_labels.train_set.dataset)
# forest_predictions = forest_reg1.predict(data.train_set.dataset)
# forest_mse = mean_squared_error(data_labels.train_set.dataset, forest_predictions)
# forest_rmse = np.sqrt(forest_mse)
# print(forest_rmse)
# forest_scores = cross_val_score(forest_reg1, data.train_set.dataset,
# data_labels.train_set.dataset, scoring='neg_mean_squared_error', cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# print("--- Random Forest Regressor RMSE Scores ---")
# display_scores(forest_rmse_scores)


def hyper_forest_stuff():
    print("--- hyper parameter tuning on forest using gridcv ---")
    start_ = time.time()
    param_grid = [
        {'n_jobs': [-1],
         'bootstrap': [False],
         'n_estimators': [150, 300],
         'max_features': [1]},  # 0.57840 False, 300, 1
        # {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                               return_train_score=True)

    grid_search.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    print("Best params: %s" % grid_search.best_params_)
    print("Best score: %s" % np.sqrt(-grid_search.best_score_))
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres['params']):
        print(np.sqrt(-mean_score), params)
    #
    # joblib.dump(grid_search, "grid_search.pkl")

    # loaded = joblib.load("grid_search.pkl")

    print("--- Forest Classifier ---")
    start_ = time.time()
    forest_clas = RandomForestClassifier()
    forest_clas.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    preds = forest_clas.predict(data.train_set.dataset)
    print("--- Forest Classifier Accuracy ---")
    print(accuracy_score(data_labels.train_set.dataset, preds))
    preds = forest_clas.predict(data.test_set.dataset)
    print(accuracy_score(data_labels.test_set.dataset, preds))

    print("--- Random Forest Classifier HyperParameter tuning ---")
    print("cv = 7, ", param_grid)
    start_ = time.time()
    for_c = RandomForestClassifier()
    grid = GridSearchCV(for_c, param_grid, cv=7, scoring='neg_mean_squared_error',
                        return_train_score=True)
    grid.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    print("Best Params: %s" % grid.best_params_)  # [False, 1, 150]
    print("Best Score: %f" % np.sqrt(-grid.best_score_))  # 0.664899
    start_ = time.time()
    preds2 = grid.predict(data.test_set.dataset)
    end_ = time.time()
    print("Predicting took %s" % datetime.timedelta(seconds=(end_ - start_)))
    preds2_accuracy = accuracy_score(data_labels.test_set.dataset, preds2)
    print("Accuracy: %f" % preds2_accuracy)

    start_ = time.time()
    param_grid_2 = [
        {'bootstrap': [True, False],
         'n_estimators': [100, 200, 300],
         'max_features': [1],
         'n_jobs': [-1]
         }]
    print("--- Random Forest Classifier HyperParameter Tuning ---")
    print("cv = 8, ", param_grid_2)
    forest_classifier = RandomForestClassifier()
    grid_search_classifier = GridSearchCV(forest_classifier, param_grid_2, cv=8,
                                          scoring='neg_mean_squared_error', return_train_score=True)
    grid_search_classifier.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    print("Best params: %s" % grid_search_classifier.best_params_)  # [True, 1, 300]
    print("Best score: %f" % np.sqrt(-grid_search_classifier.best_score_))  # 0.663840...
    start_ = time.time()
    preds3 = grid_search_classifier.predict(data.test_set.dataset)
    end_ = time.time()
    print("Predicting took %s" % datetime.timedelta(seconds=(end_ - start_)))
    preds3_accuracy = accuracy_score(data_labels.test_set.dataset, preds3)
    print("Accuracy: %f" % preds3_accuracy)


def plot_testing(train_results, test_results, train_rmses, test_rmses, times, array, fname, aname):
    # fig, ax1 = plt.subplots()
    #
    # ax1.set_xlabel(aname)
    # ax1.set_ylabel('Accuracy Score')
    #
    # line1 = ax1.plot(array, train_results, 'b', label="Train Accuracy")
    # line4 = ax1.plot(array, train_rmses, 'black', label="Train RMSE")
    # line2 = ax1.plot(array, test_results, 'r', label="Test Accuracy")
    # line5 = ax1.plot(array, test_rmses, 'green', label="Test RMSE")
    #
    # ax1.tick_params(axis='y')
    #
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Time taken (s)')
    # ax2.set_ylim([0, max(times) + 2])
    # line3 = ax2.plot(array, times, label='Time')
    # ax2.tick_params(axis='y')
    #
    # fig.tight_layout()
    # lines = line1 + line2 + line3 + line4 + line5
    # labels = [lab.get_label() for lab in lines]
    # plt.legend(lines, labels, loc='upper left')
    # plt.savefig('figs/' + fname + '-new.png')
    # plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel(aname)
    ax1.set_ylabel('Accuracy Score')

    ax1.scatter(array, train_results, c='b', label="Train Accuracy", s=2)
    ax1.scatter(array, train_rmses, c='black', label="Train RMSE", s=2)
    ax1.scatter(array, test_results, c='r', label="Test Accuracy", s=2)
    ax1.scatter(array, test_rmses, c='green', label="Test RMSE", s=2)

    ax1.tick_params(axis='y')
    plt.legend(loc='best')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Time taken (s)')
    ax2.set_ylim([0, max(times) + 2])
    ax2.scatter(array, times, label='Time', s=2)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    # labels = [lab.get_label() for lab in lines]
    # plt.legend(lines, labels, loc='upper left')
    plt.legend(loc='best')
    plt.savefig('figs/' + fname + '-new.png', dpi=1200)
    plt.show(dpi=1200)


def plot_fitting(classy):
    classy.fit(data.train_set.dataset, data_labels.train_set.dataset)

    train_pred = classy.predict(data.train_set.dataset)
    train_rmse = np.sqrt(mean_squared_error(data_labels.train_set.dataset, train_pred))
    train_accuracy = accuracy_score(data_labels.train_set.dataset, train_pred)

    test_pred = classy.predict(data.test_set.dataset)
    test_rmse = np.sqrt(mean_squared_error(data_labels.test_set.dataset, test_pred))
    test_accuracy = accuracy_score(data_labels.test_set.dataset, test_pred)
    return train_rmse, train_accuracy, test_rmse, test_accuracy


def plot_n_estimators():
    print("--- n_estimators plotting ---")
    n_estimators = [1, 4, 8, 16, 32, 64]
    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in n_estimators:
        start_ = time.time()
        forest_class = RandomForestClassifier(n_estimators=num, n_jobs=-1)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(forest_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)

        print("n_estimators %d took: %s" % (num, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, n_estimators,
                 'n_estimators', 'n_estimators')  # No super improvement over 200 compared to time
    # taken


def plot_max_depth():
    print("--- max_depth plotting ---")
    max_depths = np.linspace(1, 32, 32, endpoint=True)

    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in max_depths:
        start_ = time.time()
        forest_class = RandomForestClassifier(max_depth=num, n_jobs=-1)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(forest_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("max_depth %d took: %s" % (num, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, max_depths,
                 'max_depths', 'max_depth')  # Diminishing returns after 21/22 but increase in time


def plot_min_samples_split():
    min_samples_splits = np.linspace(0.0001, 0.001, 10, endpoint=True)

    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in min_samples_splits:
        start_ = time.time()
        forest_class = RandomForestClassifier(min_samples_split=num, n_jobs=-1)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(forest_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("min_samples_split %.4f took: %s" % (num, datetime.timedelta(seconds=(end_ -
                                                                                    start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, min_samples_splits,
                 'min_samples_splits', 'min_samples_split')  # The lower the better!


def plot_min_samples_leaf():
    # min_samples_leafs = np.linspace(0.0001, 0.002, 20, endpoint=True)
    min_samples_leafs = [0.00001, 0.0001, 0.001, 0.0015, 0.002, 0.01, 0.1, 0.5, 1]

    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in min_samples_leafs:
        start_ = time.time()
        forest_class = RandomForestClassifier(min_samples_leaf=num, n_jobs=-1)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(forest_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("min_samples_leaf %.4f took: %s" % (num, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, min_samples_leafs,
                 'min_samples_leafs', 'min_samples_leafs')  # Lower the better again


def plot_max_features():
    # max_features = np.linspace(0.01, 0.2, 10, endpoint=True)
    max_features = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5, 1, 5, 10,
                    len(data.dataset.columns)]

    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in max_features:
        start_ = time.time()
        forest_class = RandomForestClassifier(max_features=num, n_jobs=-1)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(forest_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("max_features %.4f took: %s" % (num, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, max_features,
                 'max_features', 'max_features')  # 0.18 ??


def plot_max_features_tuned():
    # max_features = np.linspace(0.01, 0.2, 10, endpoint=True)
    max_features = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3]
    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in max_features:
        start_ = time.time()
        forest_class = RandomForestClassifier(bootstrap=False, max_depth=22, max_features=num,
                                              min_samples_leaf=0.000001,
                                              min_samples_split=0.000001, n_estimators=200,
                                              n_jobs=-1)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(forest_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("min_samples_leaf %.4f took: %s" % (num, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, max_features,
                 'max_features_tuned', 'max_features')  # 26?


def compare_classifier(class_):
    # SKLEARN MAX_ERROR METRIC?????

    train = data.train_set.dataset
    train_labels = data_labels.train_set.dataset
    train_preds = class_.predict(train)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_balanced_accuracy = balanced_accuracy_score(train_labels, train_preds)
    # train_mse = mean_squared_error(train_labels, train_preds)
    # train_rmse = np.sqrt(train_mse)
    # train_max_error = max_error(train_labels, train_preds)
    # train_confusion = confusion_matrix(train_labels, train_preds)
    print("Train Accuracy:  %.2f %%" % (train_accuracy * 100.0))
    print("Train Balanced Accuracy: %.2f %%" % (train_balanced_accuracy * 100.0))
    # print("Train MSE:       %.4f" % train_mse)
    # print("Train RMSE:      %.4f" % train_rmse)
    # print("Train Max Error: %.2f" % train_max_error)
    # print(train_confusion)
    print("Train report:")
    print(classification_report(train_labels, train_preds))

    print()

    _test = data.test_set.dataset
    _test_labels = data_labels.test_set.dataset
    test_preds = class_.predict(_test)
    test_accuracy = accuracy_score(_test_labels, test_preds)
    test_balanced_accuracy = balanced_accuracy_score(_test_labels, test_preds)
    # test_mse = mean_squared_error(_test_labels, test_preds)
    # test_rmse = np.sqrt(test_mse)
    # test_max_error = max_error(_test_labels, test_preds)
    # test_confusion = confusion_matrix(_test_labels, test_preds)
    print("Test Accuracy:   %.2f%%" % (test_accuracy * 100.0))
    print("Test Balanced Accuracy: %.2f %%" % (test_balanced_accuracy * 100.0))
    # print("Test MSE:        %.4f" % test_mse)
    # print("Test RMSE:       %.4f" % test_rmse)
    # print("Test Max Error: %.2f" % test_max_error)
    # print(test_confusion)
    print("Test report:")
    print(classification_report(_test_labels, test_preds))
    """
    High recall     - the class is correctly recognised (small number of False Negatives)
    High precision  - example labelled as positive is positive (Small number of False Positives)
    High recall, low Precision  - lot of positive examples are correctly recognised (Low FN), 
                                  but lots of false negatives
    Low recall, high recision   - Miss a lot of positive examples (High FN) but predicted 
                                  positives are positive (low FP)
                                  
    https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
    """


# plot_n_estimators()
# plot_max_depth()
# plot_min_samples_split()
# plot_min_samples_leaf()
# plot_max_features()
# plot_max_features_tuned()


def tuned_rf():
    start_ = time.time()

    # rf = RandomForestClassifier(bootstrap=False, max_depth=22, max_features=0.01,
    #                             min_samples_leaf=0.000001, min_samples_split=0.000001,
    #                             n_estimators=200, n_jobs=-1, class_weight='balanced')

    rf = RandomForestClassifier(bootstrap=True, max_depth=55, max_features=135,
                                min_samples_leaf=0.0005, min_samples_split=0.0005,
                                n_estimators=500, n_jobs=-1, class_weight=None, criterion='gini')
    rf.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Time to fit: %s" % datetime.timedelta(seconds=(end_ - start_)))
    compare_classifier(rf)


tuned_rf()


def new_grid(id_):
    if id_ == 2:
        params_grid = [
            {
                'bootstrap': [True],
                'class_weight': ['balanced', None],
                'max_depth': [1, 8, 16, 22, 26, 32, 64],
                'max_features': [0.001, 0.1, 1, 5, 10, 15, len(data.dataset.columns)],
                'min_samples_leaf': [0.00001],
                'min_samples_split': [0.0005],
                'n_estimators': [1, 8, 16, 64, 200, 500, 1000],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 3:
        params_grid = [
            {
                'bootstrap': [True, False],
                'class_weight': [None],
                'max_depth': [24, 26, 28],
                'max_features': [8, 10, 12, 17],
                'min_samples_leaf': [0.00001, 0.0001],
                'min_samples_split': [0.0008, 0.001, 0.002],
                'n_estimators': [200, 800, 1000, 1500],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 4:
        params_grid = [
            {
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [25, 26, 27],
                'max_features': [6, 7, 8, 9],
                'min_samples_leaf': [0.00001, 0.00002],
                'min_samples_split': [0.002, 0.005, 0.01, 0.1],
                'n_estimators': [1000],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 5:
        params_grid = [
            {
                'bootstrap': [True, False],
                'class_weight': ['balanced', None],
                'max_depth': [26],
                'max_features': [10, 17, 32, 64, len(data.dataset.columns)],
                'min_samples_leaf': [0.00001, 0.0005, 0.001, 0.1],
                'min_samples_split': [0.0005, 0.001, 1.0],
                'n_estimators': [200, 500],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 6:
        params_grid = [
            {
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [26],
                'max_features': [10, 50, 75, 100, 150, 175, 200, len(data.dataset.columns)],
                'min_samples_leaf': [0.001],
                'min_samples_split': [0.0005],
                'n_estimators': [10, 50, 100, 200, 400, 600, 1000, 1500],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 7:
        params_grid = [
            {
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [None, 1, 5, 8, 10, 15, 20, 22, 26, 30, 50, 100, 1000],
                'max_features': [150, len(data.dataset.columns)],
                'min_samples_leaf': [0.001],
                'min_samples_split': [0.0005],
                'n_estimators': [400],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 8:
        params_grid = [
            {
                'criterion': ['gini', 'entropy'],
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [40, 45, 50, 55, 60],
                'max_features': [1, 150, len(data.dataset.columns)],
                'min_samples_leaf': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                'min_samples_split': [0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1],
                'n_estimators': [500],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 9:
        params_grid = [
            {
                'criterion': ['gini'],
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [55],
                'max_features': [100, 110, 125, 135, 150, 160, 180],
                'min_samples_leaf': [0.0005, 0.001],
                'min_samples_split': [0.0005],
                'n_estimators': [500],
                'n_jobs': [-1]
            }
        ]
    else:
        # params_grid = [
        #     {'n_jobs': [-1],
        #      'bootstrap': [False, True],
        #      'n_estimators': [200],
        #      'min_samples_split': [0.00001, 0.0001, 0.001],
        #      'min_samples_leaf': [0.00001, 0.0001, 0.001],
        #      'max_depth': [22],
        #      'max_features': [0.01, 0.05, 0.1, 0.18, 0.2, 1]
        #      }]
        # params_grid = [
        #     {
        #         'n_jobs': [-1],
        #         'bootstrap': [False, True],
        #         'n_estimators': [16, 32, 100, 200],
        #         'min_samples_split': [0.00001, 0.0005],
        #         'min_samples_leaf': [0.00001, 1],
        #         'max_depth': [22, 26, 31],
        #         'max_features': [0.001, 0.01, 0.18, 10]
        #     }
        # ]

        # params_grid = [
        #     {
        #         'bootstrap': [True],
        #         'class_weight': ['balanced', None],
        #         'max_depth': [26],
        #         'max_features': [10],
        #         'min_samples_leaf': [0.00001, 0.0005, 0.001, 0.1, 1],
        #         'min_samples_split': [0.0005, 0.001, 0.01, 0.1, 1.0],
        #         'n_estimators': [200, 500],
        #         'n_jobs': [-1]
        #     }
        # ]

        params_grid = [
            {
                'bootstrap': [True],
                'class_weight': ['balanced', None],
                'max_depth': [26],
                'max_features': [10],
                'min_samples_leaf': [0.00001],
                'min_samples_split': [0.0005],
                'n_estimators': [200, 500],
                'n_jobs': [-1]
            }
        ]

    start_ = time.time()
    rf = RandomForestClassifier()

    grid_search = GridSearchCV(rf, params_grid, cv=8, scoring='accuracy',
                               return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    print("Best params: %s" % grid_search.best_params_)
    print("Best score: %f" % grid_search.best_score_)
    compare_classifier(grid_search)
    time_ = datetime.datetime.now().strftime('%m-%d-%H.%M')
    # for param_name in list(grid_search.param_grid[0].keys()):
    #     plot_search_validation_curve(grid_search, param_name, time_=time_, type_='RF')
    #     plot_search_validation_curve(grid_search, param_name, time_=time_, type_='RF', log=True)
    plot_search_validation_curve(grid_search, time_=time_, type_='RF')
    plot_search_validation_curve(grid_search, time_=time_, type_='RF', log=True)
    return grid_search


# grid_searcher = new_grid(1)
# grid_searcher2 = new_grid(2)
# grid_searcher3 = new_grid(3)
# grid_searcher4 = new_grid(4)
# grid_searcher5 = new_grid(5)
# grid_searcher6 = new_grid(6)
# grid_searcher7 = new_grid(7)
# grid_searcher8 = new_grid(8)
# grid_searcher9 = new_grid(9)
# test = new_grid()


def randomized_search_cv_rf():
    start_ = time.time()
    rf = RandomForestClassifier()

    params_grid = [
        {
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced'],
            'max_depth': list(range(1, 33)),
            'max_features': [0.01, 0.1, 1.0, ] + list(range(2, len(data.dataset.columns) + 1)),
            'min_samples_leaf': [0.00001, 0.00002, 0.0001, 0.001, 0.01, 0.1],
            'min_samples_split': [0.002, 0.005, 0.01, 0.1],
            'n_estimators': [1, 16, 32, 64, 200, 300, 500, 700, 1000],
            'n_jobs': [-1]
        }
    ]

    k = StratifiedKFold(n_splits=7)

    randomized = RandomizedSearchCV(rf, params_grid, cv=k, scoring='accuracy',
                                    return_train_score=True, verbose=3, n_jobs=-1, n_iter=10)
    randomized.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))

    print("Best params: %s" % randomized.best_params_)
    print("Best score: %f" % randomized.best_score_)
    compare_classifier(randomized)

    return randomized


# rand = randomized_search_cv_rf()


def plot_search_validation_curve(grid, param_name, title='Validation Curve', log=None):
    df_cv_results = pd.DataFrame(grid.cv_results_)
    param_grid_df = pd.DataFrame(grid.param_grid)
    train_scores_mean = df_cv_results['mean_train_score']
    valid_scores_mean = df_cv_results['mean_test_score']
    train_scores_std = df_cv_results['std_train_score']
    valid_scores_std = df_cv_results['std_test_score']

    param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
    param_ranges = [param_grid_df[p[6:]].values.tolist()[0] for p in param_cols]
    param_ranges_length = [len(pr) for pr in param_ranges]

    train_scores_mean = np.array(train_scores_mean).reshape(*param_ranges_length)
    valid_scores_mean = np.array(valid_scores_mean).reshape(*param_ranges_length)
    train_scores_std = np.array(train_scores_std).reshape(*param_ranges_length)
    valid_scores_std = np.array(valid_scores_std).reshape(*param_ranges_length)

    param_name_idx = param_cols.index('param_{}'.format(param_name))

    slices = []
    for idx, param in enumerate(grid.best_params_):
        if idx == param_name_idx:
            slices.append(slice(None))
            continue
        best_param_val = grid.best_params_[param]
        if isinstance(param_ranges[idx], np.ndarray):
            idx_of_best_param = param_ranges[idx].tolist().index(best_param_val)
        else:
            idx_of_best_param = param_ranges[idx].index(best_param_val)
        slices.append(idx_of_best_param)

    train_scores_mean = train_scores_mean[tuple(slices)]
    valid_scores_mean = valid_scores_mean[tuple(slices)]
    train_scores_std = train_scores_std[tuple(slices)]
    valid_scores_std = valid_scores_std[tuple(slices)]

    plt.clf()

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('score')

    lw = 2

    plot_fn = plt.plot
    if log:
        plot_fn = plt.semilogx

    param_range = param_ranges[param_name_idx]
    if not isinstance(param_range[0], numbers.Number):
        param_range = [str(x) for x in param_range]
    plot_fn(param_range, train_scores_mean, label='Training Score', color='darkorange', lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean +
                     train_scores_std, alpha=0.2, color='darkorange', lw=lw)
    plot_fn(param_range, valid_scores_mean, label='CV Score', color='navy', lw=lw)
    plt.fill_between(param_range, valid_scores_mean - valid_scores_std, valid_scores_mean +
                     valid_scores_std, alpha=0.2, color='navy', lw=lw)
    plt.legend(loc='best')

    plt.savefig('figs/' + title + ' ' + param_name + '.png', dpi=1200)
    plt.show()
    plt.close()


def decision_tree_class():
    start_ = time.time()
    dt = DecisionTreeClassifier(max_depth=26, max_features=0.001, min_samples_leaf=1,
                                min_samples_split=7, random_state=42)
    dt.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Time to fit: %s" % datetime.timedelta(seconds=(end_ - start_)))
    compare_classifier(dt)


def plot_max_depth_dt():
    max_depths = np.linspace(1, 32, 32, endpoint=True)

    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in max_depths:
        start_ = time.time()
        decision_class = DecisionTreeClassifier(max_depth=num)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(decision_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("max_depths %d took: %s" % (num, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, max_depths,
                 'dt_max_depths', 'max_depths')  # 26?


def plot_criterion_dt():
    criteria = ['gini', 'entropy']

    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in criteria:
        start_ = time.time()
        decision_class = DecisionTreeClassifier(criterion=num)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(decision_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("max_depths %s took: %s" % (num, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, criteria,
                 'dt_criterion', 'criterion')  # gini


def plot_splitter_dt():
    splitters = ['best', 'random']

    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for splitter in splitters:
        start_ = time.time()
        decision_class = DecisionTreeClassifier(splitter=splitter)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(decision_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("max_depths %s took: %s" % (splitter, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, splitters,
                 'dt_splitter', 'splitter')  # time is less on random


def plot_min_samples_split_dt():
    min_samples_splits = [0.1, 0.5, 0.9, 1, 1.5, 2, 3, 4, 6, 8, 10, 20]

    train_results = []
    train_rmses = []
    test_results = []
    test_rmses = []
    times = []
    for num in min_samples_splits:
        start_ = time.time()
        decision_class = DecisionTreeClassifier(max_depth=num)
        train_rmse, train_accuracy, test_rmse, test_accuracy = plot_fitting(decision_class)
        train_rmses.append(train_rmse)
        train_results.append(train_accuracy)
        test_rmses.append(test_rmse)
        test_results.append(test_accuracy)
        end_ = time.time()
        times.append(end_ - start_)
        print("min samples split %d took: %s" % (num, datetime.timedelta(seconds=(end_ - start_))))

    plot_testing(train_results, test_results, train_rmses, test_rmses, times, min_samples_splits,
                 'dt_min_samples_splits', 'min_samples_splits')  # 8?


# decision_tree_class()


# plot_max_depth_dt()
# plot_criterion_dt()
# plot_splitter_dt()
# plot_min_samples_split_dt()


def random_search_cv_dt():
    start_ = time.time()
    params_grid = [
        {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [1, 2, 8, 26, 64, 100, 500, None],
            'min_samples_split': [0.1, 0.5, 0.9, 1.0, 2, 7, 8, 9, 10],
            'min_samples_leaf': [0.001, 0.01, 0.1, 1, 2, 8, 20],
            'max_features': [0.001, 1, 8, 17, 'sqrt', 'log2', None],
            'max_leaf_nodes': [None, 2, 5, 10, 64, 100],
            'random_state': [42],
            'class_weight': [None, 'balanced']
        }
    ]
    dt = DecisionTreeClassifier()
    k = StratifiedKFold(n_splits=10)
    rand_search = RandomizedSearchCV(dt, params_grid, cv=k, scoring='accuracy',
                                     return_train_score=True, verbose=3, n_jobs=-1, n_iter=50000)
    rand_search.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    print("Best params: %s" % rand_search.best_params_)
    print("Best score: %f" % rand_search.best_score_)
    compare_classifier(rand_search)
    return rand_search


# random_dt = random_search_cv_dt()


def hyper_dt():
    start_ = time.time()
    # params_grid = [
    #     {
    #         'splitter': ['best', 'random'],
    #         'max_depth': [1, 2, 26],
    #         'min_samples_split': [8],
    #         'min_samples_leaf': [0.1, 0.5, 1, 2, 6, 8, 10, 20],
    #         'max_features': [0.01, 0.1, 1, 'sqrt', 'log2', 2, 8, 12],
    #         'max_leaf_nodes': [None, 2, 5, 10, 100, 2000],
    #         'random_state': [42]
    #     }
    # ]

    params_grid = [
        {
            'splitter': ['best', 'random'],
            'max_depth': [1, 2, 8, 26],
            'min_samples_split': [7, 8, 9, 10],
            'min_samples_leaf': [0.001, 1, 2, 8, 20],
            'max_features': [0.001, 1, 8, 17],
            'max_leaf_nodes': [None, 2, 5, 10],
            'random_state': [42]
        }
    ]

    dt = DecisionTreeClassifier()
    k = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(dt, params_grid, cv=k, scoring='accuracy',
                               return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    print("Best params: %s" % grid_search.best_params_)
    print("Best score: %f" % grid_search.best_score_)
    compare_classifier(grid_search)
    return grid_search


# griddy = hyper_dt()


def tuned_dt():
    start_ = time.time()
    # dt = DecisionTreeClassifier(max_depth=26, max_features=17, max_leaf_nodes=None,
    #                             min_samples_leaf=20, min_samples_split=7, random_state=42,
    #                             splitter='best')
    dt = DecisionTreeClassifier(splitter='best', random_state=42, min_samples_split=9,
                                min_samples_leaf=8, max_leaf_nodes=64, max_features=17,
                                max_depth=26, criterion='gini', class_weight=None)
    dt.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Time to fit: %s" % datetime.timedelta(seconds=(end_ - start_)))
    compare_classifier(dt)


# tuned_dt()

def validation_curve_rf():
    param_name = 'max_features'
    param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    rf = RandomForestClassifier(bootstrap=True, max_depth=25,
                                min_samples_leaf=0.00001, min_samples_split=0.002,
                                n_estimators=1000, n_jobs=-1, class_weight=None)

    train_scores, test_scores = validation_curve(rf,
                                                 data.train_set.dataset,
                                                 data_labels.train_set.dataset,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 scoring='accuracy', n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.title('Validation curve with RF')
    plt.ylabel('Score')
    plt.xlabel(param_name)
    plt.plot(param_range, train_mean, label='Training Score', color='darkorange', lw=2)
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2,
                     color='darkorange', lw=2)
    plt.plot(param_range, test_mean, label='Test Score', color='navy', lw=2)
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2,
                     color='navy', lw=2)
    plt.legend(loc='best')
    plt.show()


# validation_curve_rf()
