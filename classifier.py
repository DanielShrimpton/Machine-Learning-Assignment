"""
To run this file
Make sure you have all require packages installed to the latest versions.
Requires:
- pandas
- numpy
- sklearn
- matplotlib

install them with python -m pip install <package>
or however you normally install python packages (if using virtual envs etc)

Make sure to have folder structure:
Root/
├── data/
│   ├── assessments.csv
│   ├── studentAssessment.csv
│   └── studentInfo.csv
├── figs/
└── classifier.py


by default data prep and information is run and displayed

The tuned DT and RF are run by default using their functions
To run GridSearchCV uncomment one of the lines. It will run and save and show the graphs of results

"""


import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, \
    balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import numbers
from Class import ProcessData, data_one_hot


# My Custom class to handle pandas datasets with custom, repeated functions
# class ProcessData:
#     """A class to help with making pd.DataFrames and making useful functions on them"""
#
#     def __init__(self, folder: str = None, filename: str = None, read: bool = True,
#                  data = None, tt: bool = False):
#         """An init function for the class, can be read in from file, or from passed in data"""
#         if tt:  # If the incoming data already has test and training sets
#             self.dataset = data[0].copy()
#             self.train_set = ProcessData(read=False, data=[data[1]])
#             self.test_set = ProcessData(read=False, data=[data[2]])
#         elif read:  # If not, if it needs to read from a file
#             self.file_path = os.path.join(folder, filename)
#             self.dataset = pd.read_csv(self.file_path)
#             self.train_set = None
#             self.test_set = None
#         else:  # Else just create one from the passed in data
#             self.dataset = data[0].copy()
#             self.train_set = None
#             self.test_set = None
#
#     def head(self):
#         """Function to print pd.DataFrame.head()"""
#         print(self.dataset.head())
#
#     def info(self):
#         """Function to print pd.DataFrame.info()"""
#         print(self.dataset.info())
#
#     def count(self, column: str):
#         """Function to print the value counts on certain column"""
#         print(self.dataset[column].value_counts())
#
#     def describe(self):
#         """Function to print the pd.DataFrame.describe()"""
#         print(self.dataset.describe())
#
#     def hist(self, bins: int = 100):
#         """Function to show hist of the dataset"""
#         self.dataset.hist(bins=bins)
#         plt.show()
#
#     def test_train_split(self, test_size: float = 0.2, random_state: int = 42):
#         """Function to use sklearn.train_test_split on the data and create a test and training
#         set which are new ProcessData classes so can have the same functions run on them"""
#         tr_set, te_set = train_test_split(self.dataset, test_size=test_size,
#                                           random_state=random_state)
#         self.test_set = ProcessData(read=False, data=[te_set])
#         self.train_set = ProcessData(read=False, data=[tr_set])
#
#     def label_encode(self, name: str, new_name: str, drop: bool = True):
#         """Creates a label encoder on a certain column"""
#         column = self.dataset.columns.get_loc(name)
#         label_encoder = LabelEncoder()
#         data = self.dataset.iloc[:, column]
#         stuff = label_encoder.fit_transform(np.array(data))
#         self.dataset.loc[:, new_name] = stuff
#         print(label_encoder.classes_)
#         if drop:
#             self.drop(name)
#
#     def corr(self):
#         """Function to return a correlation matrix of the dataset"""
#         return self.dataset.corr()
#
#     def drop(self, column: str, axis: int = 1, inplace: bool = True):
#         """Function to drop the specified column from the dataset, can specify if inplace or not
#         (not by default)"""
#         if inplace:
#             if self.test_set:  # Checking to see if the test/train split is already in place
#                 self.dataset.drop(column, axis=axis, inplace=inplace)
#                 self.test_set.dataset.drop(column, axis=axis, inplace=inplace)
#                 self.train_set.dataset.drop(column, axis=axis, inplace=inplace)
#             else:
#                 self.dataset.drop(column, axis=axis, inplace=inplace)
#         else:
#             if self.test_set:
#                 data = [self.dataset.drop(column, axis=axis),
#                         self.train_set.dataset.drop(column, axis=axis),
#                         self.test_set.dataset.drop(column, axis=axis)]
#                 return ProcessData(read=False, data=data, tt=True)
#             else:
#                 data = [self.dataset.drop(column, axis=axis)]
#                 return ProcessData(read=False, data=data)


start = time.time()

FOLDER = './data/'  # folder that contains the csv data files

data, data_labels = data_one_hot(FOLDER)
end = time.time()
elapsed = datetime.timedelta(seconds=(end - start))
print("Data Processing Done! Took %s" % elapsed)


def compare_classifier(class_):
    """ A function to display the metrics of the provided classifier"""
    train = data.train_set.dataset
    train_labels = data_labels.train_set.dataset
    train_preds = class_.predict(train)
    train_accuracy = accuracy_score(train_labels, train_preds)
    train_balanced_accuracy = balanced_accuracy_score(train_labels, train_preds)
    train_mse = mean_squared_error(train_labels, train_preds)
    train_rmse = np.sqrt(train_mse)
    # train_confusion = confusion_matrix(train_labels, train_preds)
    print("Train Accuracy:  %.2f %%" % (train_accuracy * 100.0))
    print("Train Balanced Accuracy: %.2f %%" % (train_balanced_accuracy * 100.0))
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
    test_balanced_accuracy = balanced_accuracy_score(_test_labels, test_preds)
    test_mse = mean_squared_error(_test_labels, test_preds)
    test_rmse = np.sqrt(test_mse)
    # test_confusion = confusion_matrix(_test_labels, test_preds)
    print("Test Accuracy:   %.2f%%" % (test_accuracy * 100.0))
    print("Test Balanced Accuracy: %.2f %%" % (test_balanced_accuracy * 100.0))
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


def tuned_rf():
    """The tuned parameters for RF"""
    start_ = time.time()

    # rf = RandomForestClassifier(bootstrap=False, max_depth=22, max_features=0.01,
    #                             min_samples_leaf=0.000001, min_samples_split=0.000001,
    #                             n_estimators=200, n_jobs=-1, class_weight='balanced')

    rf = RandomForestClassifier(bootstrap=True, max_depth=35,
                                max_features=43, min_samples_leaf=0.0005,
                                min_samples_split=0.0005, n_estimators=500, n_jobs=-1,
                                class_weight=None, criterion='gini')
    rf.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Time to fit: %s" % datetime.timedelta(seconds=(end_ - start_)))
    compare_classifier(rf)


tuned_rf()


def plot_search_validation_curve(grid, title_='Validation Curve', log=None,
                                 time_=None, type_=''):
    """Function to plot the results of a GridSearchCV"""
    df_cv_results = pd.DataFrame(grid.cv_results_)
    param_grid_df = pd.DataFrame(grid.param_grid)
    train_mean = df_cv_results['mean_train_score']
    valid_mean = df_cv_results['mean_test_score']
    train_std = df_cv_results['std_train_score']
    valid_std = df_cv_results['std_test_score']

    param_cols = [c for c in df_cv_results.columns if c[:6] == 'param_']
    param_ranges = [param_grid_df[p[6:]].values.tolist()[0] for p in param_cols]
    param_ranges_length = [len(pr) for pr in param_ranges]

    train_mean = np.array(train_mean).reshape(*param_ranges_length)
    valid_mean = np.array(valid_mean).reshape(*param_ranges_length)
    train_std = np.array(train_std).reshape(*param_ranges_length)
    valid_std = np.array(valid_std).reshape(*param_ranges_length)

    for p in param_cols:
        param_name = p[6:]
        param_name_idx = param_cols.index('param_{}'.format(param_name))
        if int(param_ranges_length[param_name_idx]) < 2:
            continue

        if not time_:
            time_ = datetime.datetime.now().strftime('%m-%d-%H.%M')
        title = time_ + ' ' + type_ + ' ' + title_ + ' ' + param_name
        if log:
            title += ' log'

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

        train_scores_mean = train_mean[tuple(slices)]
        valid_scores_mean = valid_mean[tuple(slices)]
        train_scores_std = train_std[tuple(slices)]
        valid_scores_std = valid_std[tuple(slices)]

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

        fname = 'figs/' + title + '.png'
        exists = True
        num = 1
        while exists:
            if os.path.isfile(fname):
                fname = 'figs/' + title + '-' + str(num) + '.png'
                num += 1
            else:
                exists = False
        plt.savefig(fname, dpi=1200)
        plt.show()
        plt.close()


def new_grid(id_=None):
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
    elif id_ == 10:
        params_grid = [
            {
                'criterion': ['gini'],
                'bootstrap': [True],
                'class_weight': [None, 'balanced', 'balanced_subsample', {1: 2, 2: 1, 3: 4},
                                 {1: 2.16, 2: 1, 3: 4.2}, {1: 2.2, 2: 1, 3: 5}],
                'max_depth': [55],
                'max_features': [len(data.dataset.columns)],
                'min_samples_leaf': [0.0005],
                'min_samples_split': [0.0005],
                'n_estimators': [500],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 11:
        params_grid = [
            {
                'criterion': ['gini'],
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [None, 1, 3, 8, 10, 30, 55, 80, 100],
                'max_features': [1, 5, 10, 50, 70, 100, len(data.dataset.columns)],
                'min_samples_leaf': [0.0005],
                'min_samples_split': [0.0005],
                'n_estimators': [500],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 12:
        params_grid = [
            {
                'criterion': ['gini'],
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [25, 27, 29, 30, 31, 33, 35],
                'max_features': [45, 47, 49, 50, 51, 53, 55],
                'min_samples_leaf': [0.0005],
                'min_samples_split': [0.0005],
                'n_estimators': [500],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 13:
        params_grid = [
            {
                'criterion': ['gini'],
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [34, 35, 35, 37],
                'max_features': [42, 43, 44, 45, 46, 47],
                'min_samples_leaf': [0.0005],
                'min_samples_split': [0.0005],
                'n_estimators': [500],
                'n_jobs': [-1]
            }
        ]
    elif id_ == 14:
        params_grid = [
            {
                'criterion': ['gini'],
                'bootstrap': [True],
                'class_weight': [None],
                'max_depth': [35],
                'max_features': [43],
                'min_samples_leaf': [0.000001, 0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 1],
                'min_samples_split': [0.000001, 0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.1, 1.0],
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
    k = StratifiedKFold(n_splits=8)
    grid_search = GridSearchCV(rf, params_grid, cv=k, scoring='f1_weighted',
                               return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    print("Best params: %s" % grid_search.best_params_)
    print("Best score: %f" % grid_search.best_score_)
    compare_classifier(grid_search)
    time_ = datetime.datetime.now().strftime('%m-%d-%H.%M')
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
# grid_searcher10 = new_grid(10)
# grid_searcher11 = new_grid(11)
# grid_searcher12 = new_grid(12)
# grid_searcher13 = new_grid(13)
# grid_searcher14 = new_grid(14)
# test = new_grid()


def hyper_dt():
    """Function for hyperparameter tuning DT"""
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

    # params_grid = [
    #     {
    #         'max_depth': [None, 1, 2, 8, 26, 50, 100, 200],
    #         'max_features': [0.001, 1, 8, 17, 50, 100, 150, 200, len(data.dataset.columns)],
    #         'max_leaf_nodes': [None, 2, 5, 10],
    #         'min_samples_leaf': [0.001, 1, 2, 8, 20],
    #         'min_samples_split': [0.0001, 0.0005, 0.001, 0.01, 0.1, 1.0, 5, 10, 20, 50],
    #         'random_state': [42],
    #         'splitter': ['best', 'random']
    #     }
    # ]

    params_grid = [
        {
            'max_depth': [None],
            'max_features': [len(data.dataset.columns)],
            'max_leaf_nodes': [None, 25, 50, 75, 100, 200, 500, 1000],
            'min_samples_leaf': [0.001, 20, 30, 50, 100],
            'min_samples_split': [0.01, 50, 100],
            'random_state': [42],
            'splitter': ['best']
        }
    ]

    dt = DecisionTreeClassifier()
    k = StratifiedKFold(n_splits=8)
    grid_search = GridSearchCV(dt, params_grid, cv=k, scoring='accuracy',
                               return_train_score=True, verbose=3, n_jobs=-1)
    grid_search.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Took %s" % datetime.timedelta(seconds=(end_ - start_)))
    print("Best params: %s" % grid_search.best_params_)
    print("Best score: %f" % grid_search.best_score_)
    compare_classifier(grid_search)
    time_ = datetime.datetime.now().strftime('%m-%d-%H.%M')
    plot_search_validation_curve(grid_search, time_=time_, type_='DT')
    plot_search_validation_curve(grid_search, time_=time_, type_='DT', log=True)
    return grid_search


# griddy = hyper_dt()


def tuned_dt():
    """Tuned parameters for DT"""
    start_ = time.time()
    dt = DecisionTreeClassifier(max_depth=None, max_features=len(data.dataset.columns),
                                max_leaf_nodes=None, min_samples_leaf=0.001,
                                min_samples_split=50, random_state=42, splitter='best')
    dt.fit(data.train_set.dataset, data_labels.train_set.dataset)
    end_ = time.time()
    print("Time to fit: %s" % datetime.timedelta(seconds=(end_ - start_)))
    compare_classifier(dt)


tuned_dt()
