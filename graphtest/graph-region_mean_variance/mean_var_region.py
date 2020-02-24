#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:59:19 2019

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

# dans ce programme on va calculer les moyens et variances des regions 
mask_dir='/home/ichakdi/datadeep/ADNI/AAL90/'
mask_filelist=os.listdir(mask_dir)
mask_list=[]
for a in mask_filelist:
    complete_name= mask_dir + a
    img=nib.load(complete_name)
    fdata=img.get_fdata()
    mask_list.append(fdata)

AD_dir = "/home/ichakdi/datadeep/ADNI/AD/"
AD_filelist = os.listdir(AD_dir)
AD_region_list=[]
for a in AD_filelist:
    AD_complete_filename= AD_dir + a
    img=nib.load(AD_complete_filename)
    fdata_AD=img.get_fdata()
    region=[]
    for mask in mask_list:
        mv_region=fdata_AD[(mask>0).nonzero()]
        mean_region=mv_region.mean()# calculer le moyen 
        var_region=mv_region.var() # calculer la variance 
        data=[mean_region,var_region] # mettre le moyen et la variance de la meme region dans le meme vecteur 
        region.append(data)# on met tout les veceur de la meme image dans une lise 
    AD_region_list.append(region)# c'est une liste qui contient les liste dont les vecteurs de moyen et de variance sont sauvgardees
    
    
NC_dir = "/home/ichakdi/datacold/ADNI/NC/"
NC_filelist = os.listdir(NC_dir)
NC_region_list=[]
for a in NC_filelist:
    NC_complete_filename= NC_dir + a
    img=nib.load(NC_complete_filename)
    fdata_NC=img.get_fdata()
    region=[]
    for mask in mask_list:
        mv_region=fdata_NC[(mask>0).nonzero()]
        mean_region=mv_region.mean()
        var_region=mv_region.var()
        data=[mean_region,var_region]
        region.append(data)
    NC_region_list.append(region)
    
  # sauvgarder les listes qui contien les vecteurs de moyen et variance    
pickle_out=open("AD_region_list.pickle","wb")
pickle.dump(AD_region_list,pickle_out)
pickle_out.close()

pickle_out=open("NC_region_list.pickle","wb")
pickle.dump(NC_region_list,pickle_out)
pickle_out.close()





    
            
    
    
