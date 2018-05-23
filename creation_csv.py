#!/usr/bin/python3
# -*- coding: utf8 -*-

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
#from keras.utils import np_utils 

###################################################################


#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.layers import Dropout
# from keras.layers import Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.utils import np_utils

# from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split # pour répartir les données
#from sklearn import datasets # pour importer les datasets de sickit-learn

#from matplotlib import cm

#import itertools




###### Téléchargement du fichier images et labels depuis le package tf.contrib.learn.datasets.load_dataset

#import tensorflow as tf

#mnist = tf.contrib.learn.datasets.load_dataset("mnist")

###################################################################



###### Préparation des data frame train et test enregistré dans des csv

df = pd.read_csv('./data/mnist_complet.csv', header=None)

(X_train, X_test, y_train, y_test) = train_test_split(df.iloc[:,1:], df.iloc[:,0], test_size = 0.33, random_state = 56)

pd.DataFrame(X_test).to_csv('./data/X_test.csv', sep = ',', index=False)
pd.DataFrame(X_train).to_csv('./data/X_train.csv', sep = ',', index=False)
pd.DataFrame(y_test).to_csv('./data/y_test.csv', sep = ',', index=False)
pd.DataFrame(y_train).to_csv('./data/y_train.csv', sep = ',', index=False)

###################################################################


