#!/usr/bin/python3
# -*- coding: utf8 -*-

#sed -n -e '1,1000p' X_test.csv.big.save > minst_for_png.csv


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from matplotlib import cm 

#sed -n -e '2,1001p' X_train.csv.big.save > img_nb1000.csv
#sed -n -e '2,1001p' y_train.csv.big.save > labels_nb1000.csv

#lien vers le montage du dossier googledrive
path = "/media/tostakit/Partage/google-drive/data_scientist/mnist_data/"

df_labels = pd.read_csv(path+'data/labels_nb1000.csv', header=None)
pd.DataFrame(df_labels).to_csv(path+'labels/png_labels.csv', sep = ',', index=False)

df_img = pd.read_csv(path+'data/img_nb1000.csv', header=None)
ar_img = np.asarray(df_img, np.uint8)

#plt.imshow(ar_img[0,:].reshape(28,28), cmap = cm.binary)
#plt.imshow(ar_img[0,:].reshape(28,28))
#plt.show()

n, p = np.shape(ar_img)
for i in range(n):
#for i in range(2):

    data = np.zeros((28,28,3), dtype=np.uint8)

    img = ar_img[i,:].reshape(28,28)
    
    for j in range(3):
        data[:,:,j] = img
    
    data = 255 - data
    
    mpimg.imsave(path+"pics/"+str(i)+".png", data)
#    mpimg.imsave("https:///drive.google.com/open?id=1kiVopKo8q8Y65EzlkYqa-dtE24Mmdh2s/"+str(i)+".png", data)


# img_2 = mpimg.imread("./data/resultat.png")
# plt.imshow(img_2)
# plt.show()