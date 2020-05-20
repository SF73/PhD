# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:31:11 2019

@author: sylvain.finot
"""


import numpy as np
import numpy.ma as ma
import pandas as pd
#import matplotlib.pyplot as plt
#from PIL import Image
from SEMimage import scaleSEMimage
from scipy.signal import convolve2d, savgol_filter, medfilt
from scipy import ndimage
from FileHelper import getSpectro_info
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def compareManyArrays(arrayList):
    return (np.diff(np.vstack(arrayList).reshape(len(arrayList),-1),axis=0)==0).all()  

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



def _removeSpikes(hyp,length=51):
    logger.info("Removing spikes : Warning : spectra might be tampered")
    shape = hyp.shape
    hyp = hyp.reshape(np.prod(shape[:2]),shape[-1])
    
    baseline = ndimage.percentile_filter(hyp,50,size=(1,length))
    noise = hyp-savgol_filter(hyp,length,3,axis=1)
    mask = np.abs(noise) > 2 * np.std(noise,axis=1)[:,None]    
    hyp2 = np.copy(hyp)
    hyp2[mask] = baseline[mask]

    # data = hyp[0]
    # baseline = savgol_filter(data,51, 3)
    # noise = data - baseline
    # threshold = 2.0 * np.std(noise)
    # mask = np.abs(noise) < threshold
    
    return hyp2.reshape(shape)
def loadData(paths,deadPixeltol=None,removeSpikes=True):
        basenameL = list()
        wavelenghtL = list()
        infoL = list()
        CLdataL = list()
        xlenL = list()
        ylenL = list()
        xcoordL = list()
        ycoordL = list()
        semImageL = list()
        semCartoL = list()
        if not isinstance(paths,(list,tuple,np.ndarray)):
            paths = [paths]
        #%% Prepare data
        for i in range(len(paths)):
            dirname,filename = os.path.split(paths[i])
            filename, ext = os.path.splitext(filename)
            basenameL.append(os.path.join(dirname,filename))
            hyppath = paths[i]
            specpath = basenameL[i]+'_X_axis.asc'
            semImageL.append(basenameL[i]+'_SEM image before carto.tif')
            if os.path.exists(basenameL[i]+'_SEM_carto.asc'):
                semCartoL.append(basenameL[i]+'_SEM_carto.asc')
            #data = np.loadtxt(hyppath)
            data = pd.read_csv(hyppath,delimiter='\t',header=None,dtype=np.float).to_numpy() #faster
            xlenL.append(int(data[0,0])) #number of pixel in x dir
            ylenL.append(int(data[1,0])) #number of pixel in y dir
            #spec=np.loadtxt(specpath)[:2048]
            spec=pd.read_csv(specpath,header=None,dtype=np.float,nrows=2048).to_numpy()[:,0] #faster
            wavelenghtL.append(spec)
            xcoordL.append(data[0,1:])# x's pixels of the carto
            ycoordL.append(data[1,1:])# y's pixels of the carto
            CLdataL.append(data[2:,1:])# spectra of each point
            spectroInfos = getSpectro_info(basenameL[i]+'_spectro_info.asc')
            #'Grating','Entrance slit (mm)', 'Exposure time (s)','High voltage (V)','Spot size'
            logger.info('\n %s'%spectroInfos)
            infoL.append(spectroInfos[:,1])
        
        #self.add_meta_data({"Description":str(dict(spectroInfos))})
        wavelenghtL = np.array(wavelenghtL)
        inds = wavelenghtL.argsort(axis=0)[:,0]
        CLdataL = list(np.array(CLdataL)[inds])
        xlen = np.array(xlenL)[inds]
        ylen = np.array(ylenL)[inds]
        xcoord = np.array(xcoordL)[inds]
        ycoord = np.array(ycoordL)[inds]
        wavelenght = list(wavelenghtL[inds])
        
        SameConditions = compareManyArrays(infoL) & compareManyArrays(xcoordL) & compareManyArrays(ycoordL)
        logger.info('Same Conditions : %s', SameConditions)
        
        # Filling gap spectra's gap with 0
        WaveStep = wavelenght[0][-1]-wavelenght[0][-2]
        while(len(CLdataL)>1):
            #On cherche l'index de wavelenght[1] minimisant l'écart avec wavelenght[0]
            ridx = abs(wavelenght[0]-wavelenght[1].min()).argmin()
            #On créé les longueurs d'onde manquantes dans l'écart
            patch = np.arange(wavelenght[0][ridx],wavelenght[1][0],WaveStep)
            #On assemble les 2 spectres avec le patch au milieu
            nwavelenght = np.concatenate((wavelenght[0][:ridx],patch,wavelenght[1]))
            # patch des données CL
            patch = np.zeros((len(patch),CLdataL[1].shape[1]),dtype=np.float) * np.nan#* np.min((CLdata[0],CLdata[1])) -10
            nCLdata = np.concatenate((CLdataL[0][:ridx],patch,CLdataL[1]))
            #On remplace les spectres et les données des deux premiers par le nouveau combiné
            wavelenght = [nwavelenght,*wavelenght[2:]]
            CLdataL = [nCLdata,*CLdataL[2:]]
            
        wavelenght = np.array(wavelenght)[0]
        CLdata = np.array(CLdataL)[0]
        xlen = np.array(xlenL)[0]
        ylen = np.array(ylenL)[0]
        xcoord = np.array(xcoordL)[0]
        ycoord = np.array(ycoordL)[0]
        filepath = semImageL[0]
        if len(semCartoL)>0:
            semCarto = np.loadtxt(semCartoL[0])
        else:
            semCarto = None
            trueSEM = False
        hypSpectrum = np.transpose(np.reshape(np.transpose(CLdata),(ylen,xlen ,len(wavelenght))), (0, 1, 2))
        if removeSpikes:
            hypSpectrum = _removeSpikes(hypSpectrum)
        #correct dead / wrong pixels
        if len(paths)==1 and deadPixeltol:
            if deadPixeltol<100:
                logger.debug("Cleaning deadpixel with %d sigmas tol",deadPixeltol)
                hypSpectrum, hotpixels = correct_dead_pixel(hypSpectrum,tol=deadPixeltol)
        wavelenght = ma.masked_array(wavelenght,mask=False) #Nothing is masked
        #np.resize(self.wavelenght.mask,self.hypSpectrum.shape)
        #self.hypSpectrum = ma.masked_array(self.hypSpectrum,mask=hypmask)
        hypSpectrum = ma.masked_invalid(hypSpectrum)
    
        xscale_CL,yscale_CL,acceleration,image = scaleSEMimage(filepath)
        
        newX = np.linspace(xscale_CL[int(xcoord.min())],xscale_CL[int(xcoord.max())],len(xscale_CL))
        newY = np.linspace(yscale_CL[int(ycoord.min())],yscale_CL[int(ycoord.max())],len(yscale_CL))
        X = np.linspace(np.min(newX),np.max(newX),hypSpectrum.shape[1])
        Y = np.linspace(np.min(newY),np.max(newY),hypSpectrum.shape[0])
        # logger.debug("newX")
        # logger.debug(newX)
        # logger.debug(newX.shape)
        
        # logger.debug("X")
        # logger.debug(X)
        # logger.debug(X.shape)
        #SEM image / carto
        nImage = np.array(image.crop((xcoord.min(),ycoord.min(),xcoord.max(),ycoord.max())))
        results = {}
        results["wavelenght"] = wavelenght
        results["hyper"] = hypSpectrum
        results["X"] = X
        results["Y"] = Y
        results["semImage"] = nImage
        results["semCarto"] = semCarto
        
        #return wavelenght, hypSpectrum, X, Y, nImage
        return results