import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


# Importing the data set
dataset = pd.read_csv('./data/studentInfo.csv')
print(dataset.head())
print(dataset.info())

# Print a the count of values in the column
print("\n", dataset["final_result"].value_counts())

# Print a general statistics overview of numerical data: e.g. count, min, max, mean, std, 25% etc.
print("\n", dataset.describe())

# hist gives a histogram of count vs value for each numerical data
dataset.hist(bins=100)
plt.show()


# sys.exit()
# X = dataset.iloc[:, [2, 3]].values
X = dataset.iloc[:, [8, 9, 10]]
y = dataset.iloc[:, 11]
X2 = dataset.iloc[:, [8, 9, 10, 11]]
# print(X)
# print(y)


# labelencoder_X = LabelEncoder()
# X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categorical_features=[0])
# X = onehotencoder.fit_transform(X).toarray()
X = pd.get_dummies(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
stuff = labelencoder_y.fit_transform(np.array(X2.iloc[:, 3]))
# print(stuff)
# print(X2.loc[X2['final_result'] != 'Withdrawn', 'final_result'])#;sys.exit()
X2.loc[:, 3] = stuff

print(X2)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train, X_test = train_test_split(X2, test_size=0.25, random_state=0)

grades = X_train.copy()

grades.plot(kind='scatter', x='studied_credits', y='num_of_prev_attempts', alpha=0.4,
            figsize=(10, 7), c=3, cmap=plt.get_cmap("jet"), colorbar=True)
f = plt.gcf()
cax = f.get_axes()[1]
cax.set_ylabel('Final Result')
cax.set_yticklabels(['Distinction', '',  'Fail', '', 'Pass', '', 'Withdrawn', ''])
# plt.legend()
plt.show()
