#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Dense Neural Networks for MNIST."""


import pandas as pd
import numpy as np

###### import pour affichage des résultats
from report_result import visu_img_predict, report_conf_mat
###################################################################

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.metrics import classification_report, accuracy_score



###################################################################
###### chargement des data frame train et test à partir des csv

X_test = pd.read_csv('./data/X_test.csv', header=None)
X_train = pd.read_csv('./data/X_train.csv', header=None)
y_test = pd.read_csv('./data/y_test.csv', header=None)
y_train = pd.read_csv('./data/y_train.csv', header=None)

X_train = np.asmatrix(X_train, float)
X_test = np.asmatrix(X_test, float)


##### Normalisation normaliser les pixels (de 0 à 255)
##### des données X_train et X_test afin qu'ils soient compris entre 0 et 1
X_train = X_train/255
X_test = X_test/255

##### Transformer les labels de y_train et y_test en vecteurs
##### catégorielles binaires (one hot)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

###################################################################



###### DNN méthode

num_pixels = X_train.shape[1]
num_classes = y_test.shape[1]

##### Créer un nouveau modèle séquentiel de Neural Network, appelé "model
model = Sequential()

##### Ajouter un premier fully-connected layer , de même dimension d'entrée
##### et de sortie égale au nombre de pixels, avec une fonction d'initialisation
##### gaussienne (i.e. normal) pour les poids, et une fonction d'activation relu.
model.add(Dense(output_dim = num_pixels, input_dim = num_pixels, init ='normal', activation ='relu'))

##### Ajouter un deuxième fully connected layer, avec le nombre de classe comme
##### dimension de sortie, une fonction d'initialisation gaussienne (i.e. normal)
##### pour les poids, et enfin, une fonction d'activation softmax pour renvoyer
##### une probabilité de prédiction pour chaque classe.
model.add(Dense(output_dim = num_classes, init ='normal', activation ='softmax'))

##### configurer le processus d'entraînement, avec la méthode compile,
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

##### Entraîner le modèle avec les données X_train et y_train, grâce à la méthode
##### modele.fit, avec 10 epochs, des batchs de taille 200 et rajouter "verbose = 2"
##### pour avoir un aperçu détaillé de la progression de l'entraînement, epoch par epoch.
model.fit(x = X_train , y = y_train, batch_size = 200 , epochs = 10 , verbose = 2)

test_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test)
print(" \nPerte: %.3f. Erreur: %.2f%%" % (score[0], 100-score[1]*100))



##### La fonction d'activation softmax utilisée à la fin de notre modèle, fait que
##### celui-ci retourne pour chaque image un vecteur de probabilités de longueur 10.

#####     Dans un vecteur appelé "pred" remplacer ces vecteurs de probabilités par le
#####     label avec la plus grande probailité, grâce à la méthode argmax.

#####     De la même façon, dans un vecteur appelé "y", retransformer les vecteurs
#####     one hot de y_test en labels.

#####     Avec ces deux vecteurs, afficher un compte-rendu plus détaillé de l'évaluation
#####     de la classification, grâce à la fonction classification_report.
pred = test_pred.argmax(axis=1)
y = y_test.argmax(axis=1)


###### affichage des résultats
print("Précision de la prédiction: %.2f%  % " %(accuracy_score(y, pred)*100) )
print(classification_report(y,pred))
 
visu_img_predict(X_test,y,pred)

report_conf_mat(y, pred, limite = 15)

