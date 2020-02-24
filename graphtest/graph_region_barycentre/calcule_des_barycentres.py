#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:29:31 2019

@author: ichakdi
"""



import keras
import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D,  MaxPooling3D, Add, Input, Concatenate
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras import regularizers
import IPython
import time
import math
import pickle
# lire les image de AD et NC de notre base de donnee 

mask_dir='/home/ichakdi/datadeep/ADNI/AAL90/' # le chemin de dossier des regions
mask_filelist=os.listdir(mask_dir)
mask_list=[]
for a in mask_filelist:
    complete_name= mask_dir + a
    img=nib.load(complete_name)
    fdata=img.get_fdata()
    mask_list.append(fdata)




AD_dir = "/home/ichakdi/datadeep/ADNI/AD/"
AD_filelist = os.listdir(AD_dir)
img_AD=[]
G_list_AD=[]# la liste qui va contenir les barycentres ponderes des regions des images de AD
for a in AD_filelist:
    AD_complete_filename = AD_dir + a
    img = nib.load(AD_complete_filename)
    fdata_AD = img.get_fdata()
    img_AD.append(fdata_AD)
    G_list=[]
    for mask in mask_list:
        AD_mask=fdata_AD*mask
        z1=np.nonzero(AD_mask)# c'est une liste qui contient 3 listes, chaque liste contient les cordonnees des valeurs different de zero suivant un axe  
        z2=AD_mask[z1]
        G=[]
        for k in z1:
            s=0
            for i in range(len(z2)):
                s=s+((k[i]+1)*z2[i])
                g=s/len(z2) # calcule les cordonnees des barycentre 
            G.append(g)
        G_list.append(G)
    G_list_AD.append(G_list) # la dimension de cette liste est (247,90,3)

NC_dir = "/home/ichakdi/datadeep/ADNI/NC/"
NC_filelist = os.listdir(NC_dir)
img_NC=[]
G_list_NC=[]# la liste qui va contenir les barycentres ponderes des regions des images de NC
for a in NC_filelist:
    NC_complete_filename = NC_dir + a
    img = nib.load(NC_complete_filename)
    fdata_NC = img.get_fdata()
    img_NC.append(fdata_NC)
    G_list=[]
    for mask in mask_list:
        NC_mask=fdata_NC*mask
        z1=np.nonzero(NC_mask)
        z2=NC_mask[z1]
        G=[]
        for k in z1:
            s=0
            for i in range(len(z2)):
                s=s+((k[i]+1)*z2[i])
                g=s/len(z2)
            G.append(g)
        G_list.append(G)
    G_list_NC.append(G_list) # la dimension de cette liste est (246,90,3)

    