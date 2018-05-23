#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for MNIST."""


import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from os import walk

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
import tensorflow as tf

def create_batches(batch_size, path):

    lst_labels = []
    df_labels = pd.read_csv(path+'labels/png_labels.csv', header=None)
    nb_row = len(df_labels)
    for i in range(nb_row):
        lst_labels.append(df_labels[0][i])

    lst_img = []
    for i in lst_labels:
        lst_img.append(mpimg.imread(path+'pcis/'+str(i)+'.png'))
    lst_img = np.asarray(lst_img)


    while (True):
        for i in range(0,nb_row,batch_size):
            yield(lst_img[i:i+batch_size],lst_labels[i:i+batch_size])



#lien vers le montage du dossier googledrive
path = "~/google-drive/data_scientist/mnist_data/"

# #Création de la liste des images et des labels
# lst_img = []
# for (dir, subdir, img) in walk(path+"pics/"):
#     lst_img.append(img)

# ###################################################################
# ###### chargement des images
# 
# 
# 
# ##### Normalisation normaliser les pixels (de 0 à 255)
# ##### des données X_train et X_test afin qu'ils soient compris entre 0 et 1
# X_train = X_train/255
# X_test = X_test/255
# 
# ##### Transformer les labels de y_train et y_test en vecteurs
# ##### catégorielles binaires (one hot)
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# 
# num_pixels = X_train.shape[1]
# num_classes = y_test.shape[1]
# 
# ###################################################################

imgs = tf.placeholder(tf.float32,shape=[None,height,width,colors])
lbls = tf.placeholder(tf.int32, shape=[None,label_dimension])

with tf.Session() as sess:
#define rest of graph here
# convolutions or linear layers and cost function etc.


  batch_generator = create_batches(batch_size)
  for i in range(number_of_epochs):
    images, labels = batch_generator.next()
    loss_value = sess.run([loss], feed_dict={imgs:images, lbls:labels})
    
    


###### DNN méthode
model = Sequential()
model.add(Conv2D(nb_filter= 32 , nb_row= 5 , nb_col= 5, border_mode = 'valid', input_shape= (1,28,28), activation='relu', data_format = 'channels_first'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation ='relu'))
model.add(Dense(10, activation ='softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

##### Entraîner le modèle avec les images
model.fit(x = X_train , y = y_train, batch_size = 200 , epochs = 10 , verbose = 2)
#model.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps)

test_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test)
print(" \nPerte: %.3f. Erreur: %.2f%%" % (score[0], 100-score[1]*100))

pred = test_pred.argmax(axis=1)
y = y_test.argmax(axis=1)


###### affichage des résultats
# print("Précision de la prédiction: %.2f%  % " %(accuracy_score(y, pred)*100) )
# print(classification_report(y,pred))
# 
# visu_img_non_predict(X_test,y,pred)
#  
# report_conf_mat(y, pred, limite = 10)

