# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:31:11 2019

@author: sylvain.finot
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def correct_dead_pixel(hyp,tol=3):
    """Average all the pixel which differ from the mean of its 8 neighbours more than tol times the std"""
    kernel = np.ones((3,3))
    kernel[1,1]=0
    kernel = kernel / kernel.sum()
#    hypSpectrum[:,:,i] #image totale a la longueur d'onde wavelenght[i]
    corrected = np.copy(hyp)
    integrated = np.sum(hyp,axis=2)
    neighbor_mean = convolve2d(integrated, kernel, mode='same',boundary='fill', fillvalue=0)
    neighbor_squared_mean = convolve2d(integrated**2, kernel, mode='same',boundary='fill', fillvalue=0)
    std = np.sqrt(neighbor_squared_mean-neighbor_mean**2)
    hot = (abs(integrated-neighbor_mean))>tol*std
    hotpixels = np.where(hot.flatten()==True)
    mean_hyp = np.copy(hyp)
    for i in range(hyp.shape[2]):    
        mean_hyp[:,:,i] = convolve2d(hyp[:,:,i], kernel, mode='same',boundary='fill', fillvalue=0)
    mask = hot[:,:,np.newaxis].repeat(hyp.shape[2],2)
    corrected = hyp*(1-mask)+mean_hyp*mask
#    for i in range(hyp.shape[2]):    
#        neighbor_mean = convolve2d(hyp[:,:,i], kernel, mode='same',boundary='fill', fillvalue=0)
#        neighbor_squared_mean = convolve2d(hyp[:,:,i]**2, kernel, mode='same',boundary='fill', fillvalue=0)
#        std = np.sqrt(neighbor_squared_mean-neighbor_mean**2)
#        hot = (abs(hyp[:,:,i]-neighbor_mean))>3*std
#        hot[0,:] = 0
#        hot[-1,:] = 0
#        hot[::,-1] = 0
#        hot[::,0] = 0
    return corrected, hotpixels