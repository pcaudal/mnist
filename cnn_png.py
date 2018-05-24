#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for MNIST."""


import pandas as pd
import numpy as np
import matplotlib.image as mpimg

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

    print("1 step")
    ar_labels = []
    df_labels = pd.read_csv(path+'labels/png_labels.csv', header=None)
    nb_row = len(df_labels)
    for i in range(nb_row):
        ar_labels.append(df_labels[0][i])
    ar_labels = np.asarray(ar_labels)

    print("2 step")
    ar_images = []
    for i in ar_labels:
        ar_images.append(mpimg.imread(path+'pics/'+str(i)+'.png'))
    ar_images = np.asarray(ar_images)

    print("3 step")
    while (True):
        for i in range(0,nb_row,batch_size):
            yield(ar_images[i:i+batch_size],ar_labels[i:i+batch_size])
            
    print("4 step")


#link to to images et labels repository (googledrive)
#path = "~/google-drive/data_scientist/mnist_data/"
path = "/media/tostakit/Partage/google-drive/data_scientist/mnist_data/"
# Number of Classes and Epochs of Training
nb_epoch = 10
batch_size = 2
number_of_epochs = 10
# Input Image Dimensions
img_rows, img_cols = 28, 28
# Number of Convolutional Filters to use
nb_filters = 32
# Convolution Kernel Size
#kernel_size = [5,5]
# (28 rows, 28 cols, 1 channel)
#input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(nb_filters, kernel_size = (5,5),
#                  border_mode = 'valid',
#                  padding = 'valid',
                  padding = 'valid',
                  input_shape = (img_rows, img_cols, 4),
                  activation='relu',
                  data_format = 'channels_last'))
#                  data_format = 'channels_first'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation ='relu'))
model.add(Dense(10, activation ='softmax'))
#model.add(Dense(10, kernel_initializer="uniform", activation="linear"))
model.compile(optimizer = 'adam',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

batch_generator = create_batches(batch_size, path)
for images, labels in batch_generator:
    # One hot conversion
    labels = np_utils.to_categorical(labels, 10)
#    labels = np_utils.to_categorical(labels)
    # Image normalisation for a better training model efficaciency
    images = images/255
    model.train_on_batch(images, labels)
    

# test_pred = model.predict(X_test)
# score = model.evaluate(X_test, y_test)
# print(" \nPerte: %.3f. Erreur: %.2f%%" % (score[0], 100-score[1]*100))
#  
# pred = test_pred.argmax(axis=1)
# y = y_test.argmax(axis=1)

