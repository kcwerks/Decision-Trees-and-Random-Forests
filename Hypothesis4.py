# Kyle Calabro
# Dr. Kumar
# Artificial Intelligence - Project 4
# 13 April 2018

# --------------------------- Hypothesis 4 ---------------------------

from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion

if LooseVersion(sklearn_version) < LooseVersion('0.18'):
    raise ValueError('Please use scikit-learn 0.18 or newer')

from IPython.display import Image

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from pydotplus import graph_from_dot_data

from csv import reader

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from numpy import array
import numpy as np

# To retrieve the dataset from a given csv file.

def getDataset(filename):
  file = open(filename)
  lines = reader(file)
  dataset = list(lines)
  return dataset

# To retrieve the outcome variable's column as a dictionary.

def outcomeVariable(dataset, colNo):
  classVal = [row[colNo] for row in dataset]
  unique = set(classVal)
  lookup = dict()
  for i, value in enumerate(unique):
    lookup[value] = i
  for row in dataset:
    row[colNo] = lookup[row[colNo]]
  return lookup

# To convert any given column in a given dataset to float values.

def stringToFloat(dataset, colNo):
  for row in dataset:
    row[colNo] = float(row[colNo].strip())

# The name of the file.

filename = 'Hypothesis4.csv'

dataset = getDataset(filename)

stringToFloat(dataset, 6)
stringToFloat(dataset, 9)
stringToFloat(dataset, 10)

lookup = outcomeVariable(dataset, 10)

dataset = array(dataset)

X = dataset[:, [5, 6, 9]]
y = dataset[:, [10]]

print("Class Labels:", np.unique(y))

# Building a decision tree.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5, stratify = y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

tree = DecisionTreeClassifier(criterion = 'gini',
                              max_depth = 4,
                              random_state = 5)
tree.fit(X_train, y_train)

# Output the actual decision tree as an image.

dot_data = export_graphviz(tree,
                          filled = True,
                          rounded = True,
                          class_names = ['Intro Grade <= 3.5',
                                         'Intro Grade <= 3.75',
                                         'Intro Grade > 3.75'],
                          feature_names = ['SAT Math', 'H.S. GPA', 'Major Type'],
                          out_file = None)

graph = graph_from_dot_data(dot_data)
graph.write_png('Hypothesis4.png')

# Constructing a random forest and getting the accuracies of the hypothesis.

forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=5,
                                n_jobs=2)
forest.fit(X_train, y_train.ravel())

# Calculate the accuracy of the test data.

predicted = forest.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print("Mean Accuracy for Test Data: ", accuracy)

# Calculate the accuracy of the training data.

predicted = forest.predict(X_train)
accuracy = accuracy_score(y_train, predicted)

print("Mean Accuracy for Train Data: ", accuracy)
