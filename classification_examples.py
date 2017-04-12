# -*- coding: utf-8 -*-

##############################################################################
# references
##############################################################################

# www.udemy.com/machinelearning/ - I really enjoyed this course. Take it!
# original data/code at www.superdatascience.com/machine-learning/
# en.wikipedia.org/wiki/Statistical_classification
# en.wikipedia.org/wiki/Confusion_matrix


##############################################################################
# import the libraries
##############################################################################

# look to the future if running on Python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# importing the standard libraries
import os
import sys

# import 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split as split

# importing local
sys.path.append(os.path.abspath('.'))


##############################################################################
# user defined functions
##############################################################################

def plotter(X, y, title, x_label, y_label):
    """
    A useful plotting routine for binary classifiers.
    """
    X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1,
                                   stop = X[:, 0].max() + 1,
                                   step = 0.01),
                         np.arange(start = X[:, 1].min() - 1,
                                   stop = X[:, 1].max() + 1,
                                   step = 0.01))
    plt.contourf(X1, X2,
                classifier.predict(np.array([X1.ravel(),
                                             X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75,
                cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


##############################################################################
# prepare the data: read, split, and transform
##############################################################################

# read the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = split(X, y, test_size=0.25, random_state=0)

# feature scaling: mandatory for distance-based learning, helpful in plotting
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


##############################################################################
# build the model
##############################################################################

# uncomment a classifier, and change graph titles at end...
#classifier = DTC(criterion='entropy', random_state=0)
classifier = RFC(n_estimators=10, criterion='entropy', random_state=0)
#classifier = GNB()

# fit the model and predict the test set results
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


##############################################################################
# numerical assessment
##############################################################################

# set up the formats and header for reporting
header = "{:>9s},{:>10s},{:>7s},{:>6s}"
rows = "{:>9.3f},{:>10.3f},{:>7.3f},{:>6.3f}"

# Making the Confusion Matrix and derivative metrics
cm = confusion_matrix(y_test, y_pred)
a = (cm[0,0] + cm[1,1])/np.sum(cm)     # accuracy = (TP+TN)/(TP+TN+FP+FN)
p = cm[0,0]/(cm[0,0] + cm[1,0])        # precision = TP/(TP+FP)
c = cm[0,0]/(cm[0,0] + cm[0,1])        # completeness = TP/(TP+FN)
f1 = 2*p*c/(p + c)                     # blend of precision and completeness

# report the numbers
print(header.format("accuracy", "precision", "recall", "f1"))
print(rows.format(a, p, c, f1))


##############################################################################
# visual assessment
##############################################################################

# plot description
method = "Decision Tree Classification"
x_axis = "Age"
y_axis = "Estimated Salary"

# visualising the training set results
plotter(X_train, y_train, method + " (Training set)", x_axis, y_axis)

# visualising the test set results
plotter(X_test, y_test, method + " (Test set)", x_axis, y_axis)


##############################################################################
# further work...
##############################################################################

# work with domain specialists 
# avoid overfitting
# avoid overly complicated models
# trying other classifiers
# varying (some, many, all) parameters for a given classifier
# don't specify the "random" state parameters
# n-fold cross validation

# other?

