#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for MNIST."""


import pandas as pd
import numpy as np

###### import pour affichage des résultats
from report_result import visu_img_non_predict, report_conf_mat
###################################################################

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils 

from sklearn.metrics import classification_report, accuracy_score



###################################################################
###### chargement des data frame train et test à partir des csv

X_test = pd.read_csv('./data/X_test.csv', header=None)
X_train = pd.read_csv('./data/X_train.csv', header=None)
y_test = pd.read_csv('./data/y_test.csv', header=None)
y_train = pd.read_csv('./data/y_train.csv', header=None)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

##### Transformer les données X_train et X_test en matrices à 4 dimensions
##### (nb_images, profondeur, largeur, hauteur). Chacune des images sera ainsi
##### redimensionnée au format (1,28,28).
X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))


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

##### Ajouter un premier layer de convolution, appelé Conv2D, avec les paramètres suivants:
#####     nb_filter= 32 (nombre de filtres de convolution)
#####     nb_row= 5 , nb_col= 5, (chancun des filtres à une taille de 5x5)
#####     border_mode = 'valid" (pour que les filtres ne s'appliquent qu'au
#####                            partie entières de l'image, ne dépassent pas les bords)
#####     input_shape= (1,28,28) (taille de chaque image)
#####     activation = 'relu'
#####     data_format = channel_first pour indiquer que la premiere coordonnée correspond
#####                              aux channels de l'image (par défaut Conv2D considère
#####                              qu'il s'agit de la derniere coordonnée)
model.add(Conv2D(nb_filter= 32 , nb_row= 5 , nb_col= 5, border_mode = 'valid', input_shape= (1,28,28), activation='relu', data_format = 'channels_first'))

##### Puis ajouter un layer de pooling qui renvoie le maximum appelé MaxPooling2D,
##### que l'on configure avec un pool_size de taille (2,2).
model.add(MaxPooling2D(pool_size = (2,2)))


##### Dropout est un layer de régularisation pour réduire l'overfitting, qui exclue
##### aléatoirement une fraction donnée des neurones à chaque étape de l'entraînement
##### du modèle. Ainsi il réduit la co-adaptation, empêche certains neurones de trop 
##### en influencer d'autres et favorise la généralisation de l'apprentissage.
#####     Ajouter un layer de régularisation Dropout. Ici, on choisit d'exclure 20% des inputs.
#####     Ajouter un layer appelé "Flatten", qui convertit des données matricielles en un 
#####            vecteur, pour pouvoir passer dans un layer standard, fully-connected.
#####            Cette layer ne contient pas de paramètres.
#####     Ajouter par la suite un fully-connected layer , avec 128 neurones et la
#####            fonction d'activation 'ReLu'
#####     Puis le dernier, avec 10 neurones (pour 10 classes d'output) et une fonction
#####            d'activation softmax pour renvoyer des prédictions de probabilité pour 
#####            chaque classe.
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation ='relu'))
model.add(Dense(10, activation ='softmax'))

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

visu_img_non_predict(X_test,y,pred)
 
report_conf_mat(y, pred, limite = 10)

