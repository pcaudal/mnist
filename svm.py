#!/usr/bin/python3
# -*- coding: utf8 -*-

"""SVM Estimator for MNIST."""


import pandas as pd
import numpy as np

###### import pour affichage des résultats
from report_result import visu_img_no_predict, report_conf_mat
###################################################################

###### import pour la méthode SVM
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
###################################################################


###### chargement des data frame train et test à partir des csv

X_test = pd.read_csv('./data/X_test.csv', header=None)
X_train = pd.read_csv('./data/X_train.csv', header=None)
y_test = pd.read_csv('./data/y_test.csv', header=None)
y_train = pd.read_csv('./data/y_train.csv', header=None)


X_train = np.asmatrix(X_train, float)
X_test = np.asmatrix(X_test, float)
 
y_train = np.asmatrix(y_train, int)
y_test = np.asmatrix(y_test, int)

###################################################################



###### SVM méthode

train_model = svm.SVC(kernel = 'linear')
train_model.fit(X_train, y_train)
 
test_pred = train_model.predict(X_test)
 

###### affichage des résultats

print("Précision de la prédiction: %.2f%  % " %(accuracy_score(y_test, test_pred)*100) )
print(classification_report(y_test, test_pred))
 
visu_img_no_predict(X_test,y_test,test_pred)

report_conf_mat(y_test, test_pred, limite = 36)











