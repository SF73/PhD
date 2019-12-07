# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:21:41 2019

@author: sylvain.finot
"""

import numpy as np

def R2(y,y_model):
    num = np.sum((y-y_model)**2)
#    print("Sum (y-y_model)^2 : %.4e"%num)
    denom = np.sum((y-y.mean())**2)
    return 1-num/denom

def correl(i,j,cov):
    return cov[i,j]/(cov[i,i]*cov[j,j])**0.5    

def R2adj(n,p,y,y_model):
    return 1 - (1-R2(y,y_model))*(n-1)/(n-p-1)
def rChi2(n,p,y,y_model):
    return (1/(n-p))*np.sum((y-y_model)**2/y)