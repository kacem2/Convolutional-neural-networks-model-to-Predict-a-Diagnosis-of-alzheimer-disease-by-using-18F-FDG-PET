#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:50:58 2019

@author: ichakdi
"""

def norme(l1,l2):
    s=0
    for i in range(len(l1)):
        s=s+(l1[i]-l2[i])*(l1[i]-l2[i])
    return math.sqrt(s)

alpha=0.1
weight_region_AD=[]
weight_region_NC=[]

for a in AD_region_list:
    weight_matrice_region=np.zeros(shape=(90,90))
    for i in range(90):
        for j in range(90):
            w=math.exp(alpha*norme(a[i],a[j]))
            weight_matrice_region[i,j]=w
    weight_region_AD.append(weight_matrice_region)


for a in NC_region_list:
    weight_matrice_region=np.zeros(shape=(90,90))
    for i in range(90):
        for j in range(90):
            w=math.exp(alpha*norme(a[i],a[j]))
            weight_matrice_region[i,j]=w
    weight_region_NC.append(weight_matrice_region)

