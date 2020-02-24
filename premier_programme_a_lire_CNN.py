#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:54:48 2019

@author: ichakdi
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # choisir la partie du processeur de la machine dont on va faire l'exucution du programme (cette commande ne depend pas du programme )
# ce programme a pour but de faire la classification entre Alzheimer Disease (AD) et normal Control(NC) en utilisant un reseau de neurone convolutif (CNN) 
import keras # la library qui nous permet d'utiliser les reseaux de neurone 
import numpy as np
import nibabel as nib # cette library nous permet de lire les images de TEP
import matplotlib.pyplot as plt # permet de visualiser les images 
# importer les elements qu'on aura besoin dans notre reseau de neurones 

from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D,  MaxPooling3D, Add, Input, Concatenate
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras import regularizers
import IPython
import time
# lire la base de donnee d'AD
AD_dir = "/home/ichakdi/datadeep/ADNI/AD/" # le chemin de dossier de AD
AD_filelist = os.listdir(AD_dir)
img_AD=[]
for a in AD_filelist:
    AD_complete_filename = AD_dir + a
    img = nib.load(AD_complete_filename)
    fdata_AD = img.get_fdata()# une commande pour lire les data des images 
    img_AD.append(fdata_AD) # mettre les donnees des images de AD dans la liste img_AD
# lire la base de donnee de NC
NC_dir = "/home/ichakdi/datadeep/ADNI/NC/"  #le chemin de dossier de NC
NC_filelist = os.listdir(NC_dir)
img_NC=[]
for a in NC_filelist:
    NC_complete_filename = NC_dir + a
    img = nib.load(NC_complete_filename)
    fdata_NC = img.get_fdata()
    img_NC.append(fdata_NC)# mettre les donnees des images de NC dans la liste img_NC

# preparer toute la base de donnee pour pour l'entre de notre reseaux de neurone     
ADNC=img_AD+img_NC
ADNC=np.array(ADNC)
x=np.maximum(ADNC,0)
y=np.r_[np.ones([247,1]),np.zeros([246,1])]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)# preparer la partie d'apprentissage et la partie de teste dans notre reseau de neurone 
#normalization
stdtr = np.std(x_train, axis = 0)
meantr = np.mean(x_train, axis = 0)
x_train = (x_train - meantr)/(10*stdtr + 1e-10)
x_test = (x_test - meantr)/(10*stdtr + 1e-10)

# faire un changement dans la dimension des elements d'apprentissage et de test pour qu'on puisse les utiliser dans l'entree de notre reseau de neurone 
x_train = x_train.reshape(len(x_train), 91, 109, 91, 1)
x_test = x_test.reshape(len(x_test), 91, 109, 91, 1)


layer_size=256
# le nom de dossier dans lequel on va trouver les resultats bien detailler de l'execution de notre programme 

NAME="{} conv_layer  {} layer_size   {}  dense_layer {}".format(2,layer_size,3,int(time.time())) 
tensorboard=TensorBoard(log_dir='testddddddddd2_conv_180ep_lr0.0001_8542b20all_data/{}'.format(NAME)) 

#(kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))
# maintenant on commence a construire notre reseau de neurone convolutif 
# Dans cet exemple on an fait deux couches convolutifs 
model=Sequential()
model.add(Conv3D(layer_size, (3,3,3), padding = 'same', activation='relu', use_bias = False, input_shape=(91, 109, 91, 1)))# la premiere couche  
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25)) # on ajoute ce ligne apres chaque couche pour aider le "loss" de notre programme  converger (il n'est pas forcement de faire ce ligne dans un reseau de neurone)

model.add(Conv3D(128, (3,3,3), padding = 'same', use_bias = False, activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.3))

# apres deux couches convolutifs on va faire un reseau compltement connectee  
model.add(Flatten()) # cette couche a permet de convertir l'image a un vecteur colone 
model.add(Dropout(0.35))
model.add(Dense(512, activation='relu'))
model.add(Dense(128,input_dim=128,activity_regularizer=regularizers.l1(0.01)))# dans cette couche cachee on a utilise la methode de regularization l1 
model.add(Dense(1, activation='sigmoid')) # on a utilise la fonction d'activation sigmoid parce qu'il y a une sortie binaire 
sgd = SGD(lr = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True) # ce sont les parametre de l'optimiseur 
model.compile(loss = 'binary_crossentropy', optimizer='sgd', metrics = ['accuracy']) # choisir les metrics et l'optimiseur qu'on va utiliser dans notre reseau de neurone 
teste=model.fit(x_train, y_train, epochs = 200, batch_size =1, validation_split = 0.2, shuffle = True,callbacks=[tensorboard])# ici on a pris 80% de donnees pour l'apprentissage et les 20% qui reste pour le test
accuracy = model.evaluate(x_test, y_test, batch_size = 1)

# pour lire les resultas de l'exucution de ce programme par Tensorboard, vous devez faire une commande dans le terminal 
# la commande est: # tensorbord --logdir='(vous mettez ici le chemim de l'emplacement du dossier produit par le programme)' 
# apres vous allez au internet et mettre ce lien : http://deepgsm:6006



