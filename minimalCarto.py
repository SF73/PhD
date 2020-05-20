# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:33:33 2020

@author: Sylvain.FINOT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SEMimage import scaleSEMimage
import os
from scipy.ndimage import convolve1d
# from matplotlib.widgets import Slider
from Spectrum import Spectrum

#path = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-05 - T2454 - 005K\Wire 1\Hyp-cw295nm-T005K-Slit7mm-t010ms-spot5-HV2kV_gr600.dat"
#paths = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-06 - T2453 - 005K\Wire 3\carto\Hyp-cw295nm-T005K-Slit0-5mm-t1000ms-spot2-HV2kV_gr600_zoom40000.dat"
#paths=[r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-06 - T2453 - 005K\Wire 3\carto\Hyp-cw360nm-T005K-Slit0-5mm-t1000ms-spot2-HV2kV_gr600_zoom40000.dat"]
#path=r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-06 - T2453 - 005K\Wire 3\carto\Hyp-cw425nm-T005K-Slit0-5mm-t1000ms-spot2-HV2kV_gr600_zoom40000.dat"
# paths = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-20 - T2551\005K\Wire 2\Hyp_T005K_cw340nm_spot3-5_HV5kV_gr600_slit1mm_t010ms_zoom12000"
#paths = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-20 - T2578\005K\Wire 2\Hyp_T005K_cw350nm_spot3-5_HV5kV_gr600_slit1mm_t010ms_zoom8000"
#paths = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-20 - T2575\005K\Wire 1\Hyp_T005K_cw295nm_spot3-5_HV5kV_gr600_slit1mm_t010ms_zoom12000"
# if type(paths)==str:
#     paths=[paths]
def formatdata(path):
    #%%
    dirname,filename = os.path.split(path)
    filename, ext = os.path.splitext(filename)
    hyppath = path
    path = os.path.join(dirname,filename)
    specpath = path+'_X_axis.asc'
    filepath = path+'_SEM image after carto.tif'
    data = pd.read_csv(hyppath,delimiter='\t',header=None,dtype=np.float).to_numpy()
    #%%
    xlen = int(data[0,0])
    ylen = int(data[1,0])
    wavelenght = pd.read_csv(specpath,header=None,dtype=np.float,nrows=2048).to_numpy()[:,0]
    wavelenght = wavelenght[:2048]
    xcoord = data[0,1:]
    # print(np.unique(xcoord).size)
    ycoord = data[1,1:]
    
    #nombre de pixel par ligne (pour trouver la ligne incomplete)
    pixelNumber = np.array([xcoord[ycoord==y].size for y in np.unique(ycoord)])
    completerow = np.where(pixelNumber==xlen)[0]
    print(data[2:,1:completerow.size*xlen].size)
    # print(np.unique(ycoord).size)
    CLdata = data[2:,1:] #tableau de xlen * ylen points (espace) et 2048 longueur d'onde CLdata[:,n] n = numero du spectr
    #%%
    hypSpectrum = np.transpose(np.reshape(np.transpose(CLdata),(completerow.size,xlen ,len(wavelenght))), (0, 1, 2))
    average_axis = 0 #1 on moyenne le long du fil, 0 transversalement
    linescan = np.sum(hypSpectrum,axis=average_axis)
    xscale_CL,yscale_CL,acc,image = scaleSEMimage(filepath)
    newX = np.linspace(xscale_CL[int(xcoord.min())],xscale_CL[int(xcoord.max())],len(xscale_CL))
    newY = np.linspace(yscale_CL[int(ycoord.min())],yscale_CL[int(ycoord.max())],len(yscale_CL))
    X = np.linspace(np.min(newX),np.max(newX),hypSpectrum.shape[1])
    print(min(X),max(X))
    print(min(newX),max(newX))
    Y = np.linspace(np.min(newY),np.max(newY),hypSpectrum.shape[0])
    return X,Y,hypSpectrum,wavelenght

def granularite(hypSpectrum):
    pass
    #from scipy import ndimage, misc
def energy(hypSpectrum,wavelenght,X,Y,wmin=0,wmax=1000,percentile=50):
    fig,ax = plt.subplots()
    np.histogram(np.ravel(hypSpectrum),np.arange(0,2**16))
    hist,bins = np.histogram(hypSpectrum,np.arange(200,400))
    meanBaseline = bins[np.argmax(hist)]
    sig = abs((meanBaseline-bins[np.argmin(abs(hist-max(hist)/2))])/2.3528)
    amp = max(hist)
    def gaussian(x,Amp,sig,mu):
        return Amp*np.exp(-(x-mu)**2/(2*sig**2))
    from scipy.optimize import curve_fit
    popt,_ = curve_fit(gaussian,bins[:110],hist[:110],p0=[amp,sig,meanBaseline])
    # ax.plot(bins[:-1],gaussian(bins[:-1],*popt))
    idmin = np.argmin(abs(wmin-wavelenght))
    idmax = np.argmin(abs(wmax-wavelenght))+1
    wmin = wavelenght[idmin]
    wmax = wavelenght[idmax-1]
    print(wmin,wmax)
    print(idmin,idmax)
    hypSpectrum = (hypSpectrum-(meanBaseline+3*popt[1]))/(hypSpectrum[:,:,idmin:idmax].max()-(meanBaseline+3*popt[1]))
    # hypSpectrum_bg = hypSpectrum - (meanBaseline+3*popt[1])
    # hypSpectrum_bg[hypSpectrum_bg<0] = 0
    test = np.zeros((hypSpectrum.shape[:-1]))
    intensity = hypSpectrum[:,:,idmin:idmax].sum(-1)
    for i in range(hypSpectrum.shape[0]):
        for j in range(hypSpectrum.shape[1]):
            test[i,j]=np.dot(hypSpectrum[i,j,idmin:idmax],1239.84/wavelenght[idmin:idmax])/hypSpectrum[i,j,idmin:idmax].sum()
    test = np.ma.masked_where(intensity < np.percentile(intensity,percentile), test)
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    im = ax.imshow(test,interpolation="none",aspect="auto",cmap=cmap,extent=[np.min(X),np.max(X),np.max(Y),np.min(Y)])
    Spec = Spectrum(wavelenght,np.nanmax(hypSpectrum))
    maxWave = Spec.ax.axvline(1239.84/wavelenght.mean(),c='r')
    Spec.set_limitbar(wmin,wmax)
    # cutoff = Slider(axfreq, 'Percentile', 1, 99, valinit=10, valstep=1)
    global leftpressed
    leftpressed = False
    def onmotion(event):
        global leftpressed
        if leftpressed:
            x = event.xdata
            y = event.ydata
            if ((event.inaxes is not None) and (x > X.min()) and (x <= X.max()) and 
                (y > Y.min()) and (y <= Y.max())):
                indx = np.argmin(abs(x-X))
                indy=np.argmin(abs(y-Y))
                Spec.set_y(hypSpectrum[indy,indx])
                maxWave.set_xdata([test[indy,indx]])
                Spec.update()
    
    def onclick(event):
        global leftpressed
        if event.button==1:
            leftpressed = True
            onmotion(event)
    def onrelease(event):
        global leftpressed
        if event.button==1:
            leftpressed = False
            
    # def update(val):
    #     test = np.ma.masked_where(intensity < np.percentile(intensity,80), test.data)
    # cutoff.on_changed(update)
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('motion_notify_event',onmotion)
    fig.canvas.mpl_connect('button_release_event',onrelease)
    fig.colorbar(im)
    return test
    
def diff(hypSpectrum,wavelenght,X,Y,wmin=0,wmax=1000,percentile=0):
    fig,ax = plt.subplots()
    
    np.histogram(np.ravel(hypSpectrum),np.arange(0,2**16))
    hist,bins = np.histogram(hypSpectrum,np.arange(200,400))
    meanBaseline = bins[np.argmax(hist)]
    sig = abs((meanBaseline-bins[np.argmin(abs(hist-max(hist)/2))])/2.3528)
    amp = max(hist)
    def gaussian(x,Amp,sig,mu):
        return Amp*np.exp(-(x-mu)**2/(2*sig**2))
    from scipy.optimize import curve_fit
    popt,_ = curve_fit(gaussian,bins[:110],hist[:110],p0=[amp,sig,meanBaseline])
    
    idmin = np.argmin(abs(wmin-wavelenght))
    idmax = np.argmin(abs(wmax-wavelenght))+1
    wmin = wavelenght[idmin]
    wmax = wavelenght[idmax-1]
    from scipy.signal import convolve2d
    # kernel2 = [1,-2,1]
    meankernel = [1,1,1]
    kernel2 = [0,1,-1]
    kernel1 = [-1,1,0]
    # hypSpectrum = (hypSpectrum-(meanBaseline+3*popt[1]))/(hypSpectrum[:,:,idmin:idmax].max()-(meanBaseline+3*popt[1]))
    # hypSpectrum[hypSpectrum<0]=0
    a = np.abs(convolve1d(hypSpectrum[:,:,idmin:idmax],kernel1,axis=0))\
        +np.abs(convolve1d(hypSpectrum[:,:,idmin:idmax],kernel2,axis=0))
    meana = convolve1d(hypSpectrum[:,:,idmin:idmax],meankernel,axis=0)
    
    b = np.abs(convolve1d(hypSpectrum[:,:,idmin:idmax],kernel1,axis=1))\
        +np.abs(convolve1d(hypSpectrum[:,:,idmin:idmax],kernel2,axis=1))
    meanb = convolve1d(hypSpectrum[:,:,idmin:idmax],meankernel,axis=1)

    test = (0.5*1/(idmax-idmin))*(a.sum(-1)/meana.sum(-1) + b.sum(-1)/meanb.sum(-1))
    # test = test.sum(-1)
    # test = (np.diff(hypSpectrum[:,:,idmin:idmax],axis=0)**2).sum(-1)
    # test2 = (np.diff(hypSpectrum[:,:,idmin:idmax],axis=1)**2).sum(-1)
    # test = np.sqrt(test[:,:-1] + test2[:-1,:])
    test = np.ma.masked_where(test < np.percentile(test,percentile), test)
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    ax.imshow(test,interpolation="none",aspect='auto',cmap=cmap,extent=[np.min(X),np.max(X),np.max(Y[:-1]),np.min(Y[:-1])])
    Spec = Spectrum(wavelenght,np.nanmax(hypSpectrum))
    Spec.set_limitbar(wmin,wmax)
    # maxWave = Spec.ax.axvline(1239.84/wavelenght.mean())
    
    global leftpressed
    leftpressed = False
    def onmotion(event):
        global leftpressed
        if leftpressed:
            x = event.xdata
            y = event.ydata
            if ((event.inaxes is not None) and (x > X.min()) and (x <= X.max()) and 
                (y > Y.min()) and (y <= Y.max())):
                indx = np.argmin(abs(x-X))
                indy=np.argmin(abs(y-Y))
                Spec.set_y(hypSpectrum[indy,indx])
                # maxWave.set_xdata([1239.84/test[indy,indx]])
                Spec.update()
    
    def onclick(event):
        global leftpressed
        if event.button==1:
            leftpressed = True
            onmotion(event)
    def onrelease(event):
        global leftpressed
        if event.button==1:
            leftpressed = False


    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('motion_notify_event',onmotion)
    fig.canvas.mpl_connect('button_release_event',onrelease)


# paths = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-20 - T2551\005K\Wire 2\Hyp_T005K_cw340nm_spot3-5_HV5kV_gr600_slit1mm_t010ms_zoom12000"
# paths = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-20 - T2578\005K\Wire 2\Hyp_T005K_cw350nm_spot3-5_HV5kV_gr600_slit1mm_t010ms_zoom12000"
# #paths = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-20 - T2575\005K\Wire 1\Hyp_T005K_cw295nm_spot3-5_HV5kV_gr600_slit1mm_t010ms_zoom12000"

# X,Y,hypSpectrum,wavelenght = formatdata(paths)
# # energy(hypSpectrum,wavelenght,X,Y,320,348,percentile=73)
# #diff(hypSpectrum,wavelenght,X,Y,320,348,percentile=0)


# paths = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-02-20 - T2575\005K\Wire 1\Hyp_T005K_cw295nm_spot3-5_HV5kV_gr600_slit1mm_t010ms_zoom12000"
# X,Y,hypSpectrum,wavelenght = formatdata(paths)
# energy(hypSpectrum,wavelenght,X,Y,percentile=65)
#diff(hypSpectrum,wavelenght,X,Y,percentile=0)

# Specmerged = []
# for path in paths:
#     X,Y,hypSpectrum,wavelenght = formatdata(path)
#     Specmerged.append([X,Y,hypSpectrum,wavelenght])
# for result in Specmerged:
    