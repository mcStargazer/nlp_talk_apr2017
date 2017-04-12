# -*- coding: utf-8 -*-

##############################################################################
# references
##############################################################################

# www.udemy.com/machinelearning/ - I really enjoyed this course. Take it!
# original data/code at www.superdatascience.com/machine-learning/
# en.wikipedia.org/wiki/Natural_language_processing


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

# importing 3rd party libraries
#import nltk                 # run this import and next line if stopwords
#nltk.download('stopwords')  # are not already downloaded to your computer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split as split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC

# importing local
sys.path.append(os.path.abspath('.'))


##############################################################################
# prepare the data: read and clean
##############################################################################

# read the datasets
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
common_words = set(stopwords.words('english'))  # sets are faster

# clean the text
corpus = []                                     # a list to hold the results
ps = PorterStemmer()                            # lower sparsity by stemming
for i in range(0, len(dataset['Review'])):
    #i=0; i=1; i=2
    review = dataset['Review'][i]               # get the i'th review
    review = re.sub('[^a-zA-Z]', ' ', review)   # spacify non-letters
    review = review.lower()                     # make all lowercase
    review = review.split()                     # create iteratable
    review = [ps.stem(word) for word in review  # stem the words
              if not word in common_words]      # exclude stop words
    corpus.append( ' '.join(review) )


##############################################################################
# fit and assess the model
##############################################################################

# set variables for the run
features = 1000  # number of words to keep in the model
method = "GNB"   # methods include GNB, DTC, or RFC
folds = 30       # number of cross-folds to perform
verbose = 0      # if non-zero, prints metrics for each fold

# begin reporting
print("\nUsing {} Classifier: {} features, {} folds".format(method,
                                                            features,
                                                            folds))
header = "{:>8s},{:>9s},{:>10s},{:>13s},{:>8s}"
rows = "{:8d},{:>9.3f},{:>10.3f},{:>13.3f},{:>8.3f}"
if verbose:
    print(header.format("n-fold","accuracy","precision","completeness","f1"))

# use the bag-of-words model to create X and y
cv = CountVectorizer(max_features = features)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# run across multiple folds
m = {'a':[], 'p':[], 'c':[], 'f1':[]}           # dict to hold n-fold metrics
for n in range(folds):

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = split(X, y, test_size=0.20)

    # Use any appropriate classifier.
    # Commonly: Naive Bayes, Decision Trees, and Random Forests.
    # Also: CART, C5.0, Maximum Entropy
    if method == "GNB":
        classifier = GNB()
    if method == "DTC":
        classifier = DTC(criterion='entropy', random_state=0)
    if method == "RFC":
        classifier = RFC(n_estimators=10, criterion='entropy', random_state=0)

    # fit the machine learning algorithm and predict the test set results
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # making the confusion matrix and derived metrics, and storing them
    cm = confusion_matrix(y_test, y_pred)
    a = (cm[0,0] + cm[1,1])/np.sum(cm)   # accuracy = (TP+TN)/(TP+TN+FP+FN)
    p = cm[0,0]/(cm[0,0] + cm[1,0])      # precision = TP/(TP+FP)
    c = cm[0,0]/(cm[0,0] + cm[0,1])      # completeness = TP/(TP+FN)
    f1 = 2*p*c/(p + c)                   # blend of precision and completeness
    m['a'].append(a)
    m['p'].append(p)
    m['c'].append(c)
    m['f1'].append(f1)

    # report metrics for each fold
    if verbose:
        print(rows.format(n+1, a, p, c, f1))

# report summary of metrics
print("\n          accuracy, precision, completeness,      f1")
print("  minima", rows[6:].format(min(m['a']), min(m['p']),
                                  min(m['c']), min(m['f1'])))
print("    mean", rows[6:].format(np.mean(m['a']), np.mean(m['p']),
                                  np.mean(m['c']), np.mean(m['f1'])))
print("  maxima", rows[6:].format(max(m['a']), max(m['p']),
                                  max(m['c']), max(m['f1'])))


##############################################################################
# where I am going from here...
##############################################################################

# continue exploring the parameter space balancing fit with appropriateness
# study word2vec and globe data models, other stemming algorithms
# www.udemy.com/natural-language-processing-with-deep-learning-in-python/
# www.udemy.com/data-science-natural-language-processing-in-python/

