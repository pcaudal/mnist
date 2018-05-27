#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for MNIST."""


from class_data_generator import DataGenerator
from fct_cnn_png import create_dictionary
###### import pour affichage des résultats
#from report_result import visu_img_non_predict, report_conf_mat
###################################################################

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D

#from sklearn.metrics import classification_report, accuracy_score



print("cnn_png.py : Chargement des variables d'entrée")
#link to to images et labels repository (googledrive)
#path_data = "C:\\Users\\libert\\Google Drive\\data_scientist\\mnist_data\\data\\"
path_data = "/media/tostakit/Partage/google-drive/data_scientist/mnist_data/data/"


# Input Image Dimensions
img_rows, img_cols = 28, 28
# Number of Convolutional Filters to use
batch_size = 32
# Number of classes (for mnist, 10 (0 to 9)
n_classes = 10
# Number of channel is 3 for RGB png pics
n_channels = 4
# Number of Convolutional Filters to use
nb_filters = 32
# Number of epochs
n_epochs = 1
# Thread used
multi_thread = False
# Number of thread (1 if multi_thread = False)
if multi_thread:
    n_thread = 4
else:
    n_thread = 1

# Parameters for the DataGenerator method
params = {'dim': (img_rows,img_cols),
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': n_channels,
          'shuffle': False}

print("cnn_png.py : Création des dictionnaires partition et labels")
# Datasets
partition, labels = create_dictionary(path = path_data)
    

print("cnn_png.py : Création des objets training_generator et validation_generator avecc la class_data_generator")
# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)


print("cnn_png.py : Préparation du model")
#Design model
model = Sequential()
model.add(Conv2D(nb_filters, kernel_size = (5,5),
#                  border_mode = 'valid',
#                  padding = 'valid',
                  padding = 'valid',
                  input_shape = (img_rows, img_cols, n_channels),
                  activation='relu',
                  data_format = 'channels_last'))
#                  data_format = 'channels_first'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation ='relu'))
model.add(Dense(n_classes, activation ='softmax'))
#model.add(Dense(10, kernel_initializer="uniform", activation="linear"))
model.compile(optimizer = 'adam',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])


print("cnn_png.py : Modele fit")
# Train model on dataset
model.fit_generator(generator = training_generator,
                    validation_data= validation_generator,
                    use_multiprocessing = multi_thread,
                    workers = n_thread,
                    epochs = n_epochs,
                    verbose = 2)
 
print("cnn_png.py : Modele Predict")
predict = model.predict_generator(generator= validation_generator,
                                 use_multiprocessing = multi_thread,
                                 workers = n_thread,
                                 verbose = 1)
 
print("cnn_png.py : Modele evaluate")
scores = model.evaluate_generator(generator= validation_generator,
                         use_multiprocessing = multi_thread,
                         workers = n_thread)
#                        workers = n_thread,
#                        verbose = 0)

