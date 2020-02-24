#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:16:40 2019

@author: ichakdi
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

AD_region_list=pickle.load(open("AD_region_list.pickle","rb"))
NC_region_list=pickle.load(open("NC_region_list.pickle","rb"))


# une fonction qui calcule la norme entre deux vecteurs

def norme(l1,l2):
    s=0
    for i in range(len(l1)):
        s=s+(l1[i]-l2[i])*(l1[i]-l2[i])
    return math.sqrt(s)

alpha=0.1
weight_region_AD=[]
weight_region_NC=[]

for a in AD_region_list:
    weight_matrice_region=np.zeros(shape=(90,90))# matrice de dim=(90,90) qui contient que des zeros
    for i in range(90):
        for j in range(90):
            w=math.exp(alpha*norme(a[i],a[j]))
            weight_matrice_region[i,j]=w  # la matrice weight_matrice_region va contenir les poids des graphes entre les region 
    weight_region_AD.append(weight_matrice_region)


for a in NC_region_list:
    weight_matrice_region=np.zeros(shape=(90,90))
    for i in range(90):
        for j in range(90):
            w=math.exp(alpha*norme(a[i],a[j]))
            weight_matrice_region[i,j]=w
    weight_region_NC.append(weight_matrice_region)    
ADNC=weight_region_AD + weight_region_NC
ADNC=np.array(ADNC)
x=np.maximum(ADNC,0)
y=np.r_[np.ones([247,1]),np.zeros([246,1])]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


#normalization
stdtr = np.std(x_train, axis = 0)
meantr = np.mean(x_train, axis = 0)
x_train = (x_train - meantr)/(10*stdtr + 1e-10)
x_test = (x_test - meantr)/(10*stdtr + 1e-10)

x_train = x_train.reshape(len(x_train),90,90, 1)
x_test = x_test.reshape(len(x_test),90,90, 1)




 



