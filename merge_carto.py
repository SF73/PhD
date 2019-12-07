# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:15:00 2019

@author: sylvain.finot
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path
import time
import scipy.constants as cst
eV_To_nm = cst.c*cst.h/cst.e*1E9
from detect_dead_pixels import correct_dead_pixel
from SEMimage import scaleSEMimage
plt.style.use('Rapport')
plt.rc('axes', labelsize=16)

def merge(path,save=False,autoclose=False,log=False,threshold=0,deadPixeltol = 100,aspect="auto",EnergyRange = []):
    #path = [r'F:\data\2019-05-28 - T2597 - 300K\HYP-T2597-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t010ms-cw440nm/Hyp.dat',r'F:\data\2019-05-28 - T2597 - 300K\HYP-T2597-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t010ms-cw400nm/Hyp.dat']
#    path1=r'F:\data\2019-05-28 - T2597 - 300K\HYP-T2597-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t010ms-cw460nm/Hyp.dat'
#    path2=r'F:\data\2019-05-28 - T2597 - 300K\HYP-T2597-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t010ms-cw360nm/Hyp.dat'
#    
    path = [r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-03-22 - T2581 - 005K\Fil 1\HYP1-T2581-005K-Vacc5kV-spot7-zoom4000x-gr600-slit0-2-t5ms-cw460nm\Hyp.dat",r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-03-22 - T2581 - 005K\Fil 1\HYP1-T2581-005K-Vacc5kV-spot7-zoom4000x-gr600-slit0-2-t5ms-cw540nm\Hyp.dat"]
    dirname = list()
    wavelenght = list()
    CLdata = list()
    xlen = list()
    ylen = list()
    xcoord = list()
    ycoord = list()
    filepath = list()
    for i in range(len(path)):
        dirname.append(os.path.dirname(path[i]))
        hyppath = path[i]
        specpath = os.path.join(dirname[i],'Hyp_X_axis.asc')
        filepath.append(os.path.join(dirname[i],'Hyp_SEM image after carto.tif'))
        data = np.loadtxt(hyppath)
        xlen.append(int(data[0,0]))
        ylen.append(int(data[1,0]))
        wavelenght.append(np.loadtxt(specpath)[:2048])
#        wavelenght1 = wavelenght1[:2048]
        xcoord.append(data[0,1:])
        ycoord.append(data[1,1:])
        CLdata.append(data[2:,1:])
        
    wavelenght = np.array(wavelenght)
    inds = wavelenght.argsort(axis=0)[:,0]
    CLdata = list(np.array(CLdata)[inds])
    xlen = np.array(xlen)[inds]
    ylen = np.array(ylen)[inds]
    xcoord = np.array(xcoord)[inds]
    ycoord = np.array(ycoord)[inds]
    wavelenght = list(wavelenght[inds])
    
    #on suppose que deltaLambda est identique partout
    WaveStep = wavelenght[0][-1]-wavelenght[0][-2]
    while(len(CLdata)>1):
        ridx = abs(wavelenght[0]-wavelenght[1].min()).argmin()
        patch = np.arange(wavelenght[0][ridx],wavelenght[1][0],WaveStep)
        nwavelenght = np.concatenate((wavelenght[0][:ridx],patch,wavelenght[1]))
        patch = np.zeros((len(patch),CLdata[1].shape[1])) #* np.min((CLdata[0],CLdata[1])) -10
        nCLdata = np.concatenate((CLdata[0][:ridx],patch,CLdata[1]))
        wavelenght = [nwavelenght,*wavelenght[2:]]
        CLdata = [nCLdata,*CLdata[2:]]
    
    wavelenght = np.array(wavelenght)[0]
    CLdata = np.array(CLdata)[0]
    hypSpectrum = np.transpose(np.reshape(np.transpose(CLdata),(ylen[0],xlen[0] ,len(wavelenght))), (0, 1, 2))
    hypSpectrum, hotpixels = correct_dead_pixel(hypSpectrum,tol=deadPixeltol)
    

    if len(EnergyRange)==2:
        EnergyRange = 1239.842/np.array(EnergyRange)
        EnergyRange.sort()
        lidx = np.argmin(np.abs(wavelenght-EnergyRange[0]))
        ridx = np.argmin(np.abs(wavelenght-EnergyRange[1]))
        wavelenght = wavelenght[lidx:ridx]
        hypSpectrum = hypSpectrum[:,:,lidx:ridx]
    
    
    average_axis = 0 #1 on moyenne le long du fil, 0 transversalement
    linescan = np.sum(hypSpectrum,axis=average_axis)
    linescan -= linescan.min()
#    linescan = np.log10(linescan)
    xscale_CL,yscale_CL,acc,image = scaleSEMimage(filepath[0])
    fig,(ax,bx,cx)=plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [1,1, 3]})
    fig.patch.set_alpha(0) #Transparency style
    fig.subplots_adjust(top=0.9,bottom=0.12,left=0.15,right=0.82,hspace=0.1,wspace=0.05)
    newX = np.linspace(xscale_CL[int(xcoord[0].min())],xscale_CL[int(xcoord[0].max())],len(xscale_CL))
    newY = np.linspace(yscale_CL[int(ycoord[0].min())],yscale_CL[int(ycoord[0].max())],len(yscale_CL))
    nImage = np.array(image.crop((xcoord[0].min(),ycoord[0].min(),xcoord[0].max(),ycoord[0].max())))#-np.min(image)
    ax.imshow(nImage,cmap='gray',vmin=0,vmax=65535,extent=[np.min(newX),np.max(newX),np.max(newY),np.min(newY)])
    ax.set_ylabel("distance (µm)")
    hypSpectrum = hypSpectrum-threshold
    hypSpectrum = hypSpectrum*(hypSpectrum>=0)
    hypimage=np.sum(hypSpectrum,axis=2)
    hypimage -= hypimage.min()
    if log:
        hypimage=np.log10(hypimage+1)
        linescan = np.log10(linescan+1)
    lumimage = bx.imshow(hypimage,cmap='jet',extent=[np.min(newX),np.max(newX),np.max(newY),np.min(newY)])
    if average_axis==1:
        extent = [1239.842/wavelenght.max(),1239.842/wavelenght.min(),np.max(newY),np.min(newY)]
        im=bx.imshow(linescan,cmap='jet',extent=extent)
        bx.set_xlabel("energy (eV)")
        bx.set_ylabel("distance (µm)")
    else:
        extent = [np.min(newX),np.max(newX),1239.842/wavelenght.max(),1239.842/wavelenght.min()]
        im=cx.imshow(linescan.T,cmap='jet',extent=extent)
        cx.set_ylabel("Energy (eV)")
        cx.set_xlabel("distance (µm)")

    cx.set_aspect('auto')
    bx.set_aspect(aspect)
    ax.set_aspect(aspect)
    ax.get_shared_y_axes().join(ax, bx)
    pos = cx.get_position().bounds
    cbar_ax = fig.add_axes([0.85, pos[1], 0.05, pos[-1]*0.9])  
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    
    pos = bx.get_position().bounds
    cbar_ax = fig.add_axes([0.85, pos[1], 0.05, pos[-1]*0.9])
    fig.colorbar(lumimage,cax=cbar_ax)
    cbar_ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    if save==True:
        savedir = os.path.join(dirname,'Saved')
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        fig.savefig(os.path.join(dirname,'Saved','SEM+Hyp+Linescan_merged.png'),dpi=300)
    if autoclose==True:
        plt.close(fig)