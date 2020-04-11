import datetime
import time
from Class import ProcessData
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, \
    classification_report
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D

start = time.time()

FOLDER = './data/'  # folder that contains the csv data files


def first_attempt():
    studentInfo = ProcessData(folder=FOLDER, filename='studentInfo.csv', read=True)
    studentInfo.label_encode('final_result', 'result')
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
                                                   studentInfo.dataset], sort=True)])
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
    results_mini = [studentInfo.dataset.final_result.iat[x] for x in
                    range(len(studentInfo.dataset.id_student)) if
                    studentInfo.dataset.id_student.iat[x] == num]
    print(results, results_mini)


# TODO Properly process data so is unique per final result and course ID and student ID??
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
    length = len(_data.dataset.final_result)
    # for i in range(length):
    #     if data.dataset.final_result.iat[i] == 'Withdrawn':
    #         data.dataset.result.iat[i] = 0
    #     elif data.dataset.final_result.iat[i] == 'Fail':
    #         data.dataset.result.iat[i] = 1
    #     elif data.dataset.final_result.iat[i] == 'Pass':
    #         data.dataset.result.iat[i] = 2
    #     elif data.dataset.final_result.iat[i] == 'Distinction':
    #         data.dataset.result.iat[i] = 3
    _data.drop('final_result')
    _data.drop('code_module')
    _data.drop('code_presentation')
    _data.label_encode('region', 'region_')
    _data.drop('region')
    _data.label_encode('imd_band', 'imd_band_')
    _data.drop('imd_band')
    _data.label_encode('age_band', 'age_band_')
    _data.drop('age_band')
    _data.label_encode('disability', 'disability_')
    _data.drop('disability')
    _data.label_encode('gender', 'gender_')
    _data.drop('gender')
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
    _data.drop('highest_education')
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


data, data_labels = process_data()
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
# forest_scores = cross_val_score(forest_reg1, data.train_set.dataset, data_labels.train_set.dataset,
#                                 scoring='neg_mean_squared_error', cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# print("--- Random Forest Regressor RMSE Scores ---")
# display_scores(forest_rmse_scores)


def hyper_forest_stuff():
    print("--- hyper parameter tuning on forest using gridcv ---")
    start = time.time()
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
    end = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end - start)))
    print("Best params: %s" % grid_search.best_params_)
    print("Best score: %s" % np.sqrt(-grid_search.best_score_))
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres['params']):
        print(np.sqrt(-mean_score), params)
    #
    # joblib.dump(grid_search, "grid_search.pkl")

    # loaded = joblib.load("grid_search.pkl")

    print("--- Forest Classifier ---")
    start = time.time()
    forest_clas = RandomForestClassifier()
    forest_clas.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end - start)))
    preds = forest_clas.predict(data.train_set.dataset)
    print("--- Forest Classifier Accuracy ---")
    print(accuracy_score(data_labels.train_set.dataset, preds))
    preds = forest_clas.predict(data.test_set.dataset)
    print(accuracy_score(data_labels.test_set.dataset, preds))

    print("--- Random Forest Classifier HyperParameter tuning ---")
    print("cv = 7, ", param_grid)
    start = time.time()
    for_c = RandomForestClassifier()
    grid = GridSearchCV(for_c, param_grid, cv=7, scoring='neg_mean_squared_error',
                        return_train_score=True)
    grid.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end - start)))
    print("Best Params: %s" % grid.best_params_)  # [False, 1, 150]
    print("Best Score: %f" % np.sqrt(-grid.best_score_))  # 0.664899
    start = time.time()
    preds2 = grid.predict(data.test_set.dataset)
    end = time.time()
    print("Predicting took %s" % datetime.timedelta(seconds=(end - start)))
    preds2_accuracy = accuracy_score(data_labels.test_set.dataset, preds2)
    print("Accuracy: %f" % preds2_accuracy)

    start = time.time()
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
    end = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end - start)))
    print("Best params: %s" % grid_search_classifier.best_params_)  # [True, 1, 300]
    print("Best score: %f" % np.sqrt(-grid_search_classifier.best_score_))  # 0.663840...
    start = time.time()
    preds3 = grid_search_classifier.predict(data.test_set.dataset)
    end = time.time()
    print("Predicting took %s" % datetime.timedelta(seconds=(end - start)))
    preds3_accuracy = accuracy_score(data_labels.test_set.dataset, preds3)
    print("Accuracy: %f" % preds3_accuracy)


def plot_testing(train_results, test_results, train_rmses, test_rmses, times, array, fname, aname):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel(aname)
    ax1.set_ylabel('Accuracy Score')

    line1 = ax1.plot(array, train_results, 'b', label="Train Accuracy")
    line4 = ax1.plot(array, train_rmses, 'black', label="Train RMSE")
    line2 = ax1.plot(array, test_results, 'r', label="Test Accuracy")
    line5 = ax1.plot(array, test_rmses, 'green', label="Test RMSE")

    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Time taken (s)')
    ax2.set_ylim([0, max(times) + 2])
    line3 = ax2.plot(array, times, label='Time')
    ax2.tick_params(axis='y')

    fig.tight_layout()
    lines = line1 + line2 + line3 + line4 + line5
    labels = [lab.get_label() for lab in lines]
    plt.legend(lines, labels, loc='upper left')
    plt.savefig('figs/' + fname + '.png')
    plt.show()


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
    n_estimators = [32, 64, 100, 200, 300, 400, 500]
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
    min_samples_leafs = np.linspace(0.0001, 0.002, 20, endpoint=True)

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
    max_features = np.linspace(0.01, 0.2, 10, endpoint=True)

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
    train_mse = mean_squared_error(train_labels, train_preds)
    train_rmse = np.sqrt(train_mse)
    train_confusion = confusion_matrix(train_labels, train_preds)
    print("Train Accuracy:  %.2f %%" % (train_accuracy * 100.0))
    print("Train MSE:       %.4f" % train_mse)
    print("Train RMSE:      %.4f" % train_rmse)
    # print(train_confusion)
    print("Train report:")
    print(classification_report(train_labels, train_preds))

    print()

    _test = data.test_set.dataset
    _test_labels = data_labels.test_set.dataset
    test_preds = class_.predict(_test)
    test_accuracy = accuracy_score(_test_labels, test_preds)
    test_mse = mean_squared_error(_test_labels, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_confusion = confusion_matrix(_test_labels, test_preds)
    print("Test Accuracy:   %.2f%%" % (test_accuracy * 100.0))
    print("Test MSE:        %.4f" % test_mse)
    print("Test RMSE:       %.4f" % test_rmse)
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
    start = time.time()

    rf = RandomForestClassifier(bootstrap=False, max_depth=22, max_features=0.01,
                                min_samples_leaf=0.000001, min_samples_split=0.000001,
                                n_estimators=200, n_jobs=-1)
    rf.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end = time.time()
    print("Time to fit: %s" % datetime.timedelta(seconds=(end - start)))
    compare_classifier(rf)


tuned_rf()


def decision_tree_class():
    start = time.time()
    dt = DecisionTreeClassifier(max_depth=26, max_features=0.001, min_samples_leaf=1,
                                min_samples_split=7, random_state=42)
    dt.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end = time.time()
    print("Time to fit: %s" % datetime.timedelta(seconds=(end - start)))
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


decision_tree_class()


# plot_max_depth_dt()
# plot_criterion_dt()
# plot_splitter_dt()
# plot_min_samples_split_dt()


def hyper_dt():
    start = time.time()
    params_grid = [
        {
            'splitter': ['best', 'random'],
            'max_depth': [1, 2, 26],
            'min_samples_split': [8],
            'min_samples_leaf': [0.1, 0.5, 1, 2, 6, 8, 10, 20],
            'max_features': [0.01, 0.1, 1, 'sqrt', 'log2', 2, 8, 12],
            'max_leaf_nodes': [None, 2, 5, 10, 100, 2000],
            'random_state': [42]
        }
    ]

    params_grid = [
        {
            'splitter': ['best', 'random'],
            'max_depth': [24, 25, 26, 27, 28],
            'min_samples_split': [7, 8, 9, 10],
            'min_samples_leaf': [1],
            'max_features': [0.001, 1, 12],
            'max_leaf_nodes': [None],
            'random_state': [42]
        }
    ]

    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(dt, params_grid, cv=10, scoring='accuracy',
                               return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end - start)))
    print("Best params: %s" % grid_search.best_params_)
    print("Best score: %f" % grid_search.best_score_)
    compare_classifier(grid_search)


# hyper_dt()


def new_grid():
    start = time.time()
    rf = RandomForestClassifier()
    params_grid = [
        {'n_jobs': [-1],
         'bootstrap': [False, True],
         'n_estimators': [200],
         'min_samples_split': [0.00001, 0.0001, 0.001],
         'min_samples_leaf': [0.00001, 0.0001, 0.001],
         'max_depth': [22],
         'max_features': [0.01, 0.05, 0.1, 0.18, 0.2, 1]
         }]
    params_grid = [
        {
            'n_jobs': [-1],
            'bootstrap': [False, True],
            'n_estimators': [16, 100, 200],
            'min_samples_split': [0.00001],
            'min_samples_leaf': [0.00001],
            'max_depth': [22],
            'max_features': [0.001, 0.01, 0.18]
        }
    ]
    grid_search = GridSearchCV(rf, params_grid, cv=10, scoring='accuracy',
                               return_train_score=True, verbose=3)
    grid_search.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end - start)))
    print("Best params: %s" % grid_search.best_params_)
    print("Best score: %f" % grid_search.best_score_)
    compare_classifier(grid_search)
