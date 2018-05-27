#!/usr/bin/python3
# -*- coding: utf8 -*-

"""Convolutional Neural Networks for MNIST."""


from numpy import genfromtxt

def create_dictionary(path = 'data/', ratio = 0.2):

    print("fct create_dictionary : Début de la création de dict_labels")
    #labels
    #{'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
    list_values = list(map(int, genfromtxt(path+'png_labels.csv', delimiter=',')))
    nb_img = len(list_values)
    list_keys = list(map(str, range(nb_img)))
    dict_labels = dict(zip(list_keys,list_values))
    print("fct create_dictionary : dict_labels est créé")
      
    print("fct create_dictionary : Début de la création de dict_partition")
    #partition
    #{'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
    nb_img_test = round(ratio*nb_img)
    nb_img_train = nb_img - nb_img_test
    
    dict_partition = dict({'train':list(map(str, range(nb_img_train))), 'validation':list(map(str, range(nb_img_train, nb_img_train+nb_img_test)))})
    print("fct create_dictionary : dict_partition est créé")
    
    return dict_partition, dict_labels
