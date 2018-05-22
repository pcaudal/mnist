#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm 

import itertools
from sklearn.metrics import confusion_matrix


def visu_img_no_predict(X_test,y_test,test_pred):
    
    img_nb=0
    
    for i in np.random.choice(np.arange(0, len(y_test)), size=6):
        img_nb += 1
        img = X_test[i,:] 
        img = img.reshape(28,28)
         
        plt.subplot(2,3,img_nb)
        plt.axis('off')
        plt.imshow(img,cmap = cm.binary)
        plt.title('Prediction: %i' % test_pred[i])
     
        plt.show()


def report_conf_mat(y, pred, limite):
 
    cnf_matrix = confusion_matrix(y, pred)
    
    ###Optionnel: Afficher une matrice de confusion sous forme de tableau coloré
    classes= range(0,10)
    
    plt.figure()
    
    plt.imshow(cnf_matrix, interpolation='nearest',cmap='Blues')
    plt.title("Matrice de confusion")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > ( cnf_matrix.max() / 2) else "black")
    
    plt.ylabel('Vrais labels')
    plt.xlabel('Labels prédits')
    plt.show()


    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        if(cnf_matrix[i,j] > limite and i!=j):
            print("Le chiffre {0} a été pris pour le chiffre {1}".format(i,j))

###################################################################