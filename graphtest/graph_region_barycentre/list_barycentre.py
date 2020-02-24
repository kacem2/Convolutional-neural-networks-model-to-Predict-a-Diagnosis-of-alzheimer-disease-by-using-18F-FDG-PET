#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:58:47 2019

@author: ichakdi
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,  MaxPooling3D, Add, Input, Concatenate, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras import regularizers
import IPython
import time
import math
import pickle

def norme(l1,l2): # une fonction qui calcule la norme geometrique entre deux vecteurs 

    s=0
    for i in range(len(l1)):
        s=s+(l1[i]-l2[i])*(l1[i]-l2[i])
    return math.sqrt(s)
alpha=0.1
weight_matric_list_AD=[] # cette liste va contenir les matices dans lequel on met des pois selon relation  exp(alpha*norme(a[i],a[j]))
weight_matric_list_NC=[] # la meme chose pour cette liste 
# lire les donnees qui sont deja sauvgarder dans deux fichiers par le premier programme
G_AD_list=pickle.load(open("AD.pickle","rb"))
G_NC_list=pickle.load(open("NC.pickle","rb"))

# la boucle qui va calculer les poids selon les cordonnees des barycentres des regions 
for a in G_AD_list:
    weight_matric=np.zeros(shape=(90,90))
    for i in range(90):
        for j in range(90):
            w=math.exp(alpha*norme(a[i],a[j]))
            weight_matric[i][j]=w # les poids qu'on va mettre dans la matrice weight_matric_list_AD
    
    weight_matric_list_AD.append(weight_matric)
    
    
for a in G_NC_list:
    weight_matric=np.zeros(shape=(90,90))
    for i in range(90):
        for j in range(90):
            w=math.exp(alpha*norme(a[i],a[j]))
            weight_matric[i][j]=w# les poids qu'on va mettre dans la matrice weight_matric_list_NC
    
    weight_matric_list_NC.append(weight_matric)

ADNC=weight_matric_list_AD+weight_matric_list_NC
ADNC=np.array(ADNC)
x=np.maximum(ADNC,0)
y=np.r_[np.ones([247,1]),np.zeros([246,1])]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

#normalization 
stdtr = np.std(x_train, axis = 0)
meantr = np.mean(x_train, axis = 0)
x_train = (x_train - meantr)/(10*stdtr + 1e-10)
x_test = (x_test - meantr)/(10*stdtr + 1e-10)

x_train = x_train.reshape(len(x_train),90,90, 1)
x_test = x_test.reshape(len(x_test),90,90, 1)
# notre reseau de neurone convolutif 2D
model=Sequential()

model.add(Conv2D(1024, (3,3), activation='relu', input_shape=(90,90, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'binary_crossentropy', optimizer='sgd', metrics = ['accuracy'])
teste=model.fit(x_train, y_train, epochs = 2, batch_size =1, validation_split = 0.2, shuffle = True)
accuracy = model.evaluate(x_test, y_test, batch_size = 1)
 







    

