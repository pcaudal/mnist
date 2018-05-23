#!/usr/bin/python3
# -*- coding: utf8 -*-

"""SVM Estimator for MNIST."""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm 


###### import pour la méthode SVM

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools

###################################################################


###### chargement des data frame train et test à partir des csv

X_test = pd.read_csv('./data/X_test.csv', header=None)
X_train = pd.read_csv('./data/X_train.csv', header=None)
y_test = pd.read_csv('./data/y_test.csv', header=None)
y_train = pd.read_csv('./data/y_train.csv', header=None)


X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

#y_train = np.array(y).astype(int)
#y_train = np.array(y).astype(int)

###################################################################

