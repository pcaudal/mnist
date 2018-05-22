#!/usr/bin/python3
# -*- coding: utf8 -*-

###### Téléchargement du fichier images et labels depuis le package
###### tf.contrib.learn.datasets.load_dataset au format ubyte.gz

import tensorflow as tf

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

###### après téléchargement extraire les fichier gz et les convertir en csv
###### avec le script conversion.py
###################################################################
