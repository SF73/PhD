# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:46:34 2019

@author: Sylvain
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:57:46 2019

@author: sylvain.finot
"""

import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os.path
import argparse
import time
import pandas as pd
import scipy.constants as cst
eV_To_nm = cst.c*cst.h/cst.e*1E9
from detect_dead_pixels import correct_dead_pixel
from SEMimage import scaleSEMimage
from FileHelper import getListOfFiles
from matplotlib.widgets import MultiCursor, Cursor, SpanSelector
import logging
logger = logging.getLogger('__name__')
plt.style.use('Rapport')
plt.rc('axes', labelsize=16)
                
class MyMultiCursor(MultiCursor):
    def __init__(self, canvas, axes, useblit=True, horizOn=[], vertOn=[], **lineprops):
        super(MyMultiCursor, self).__init__(canvas, axes, useblit=useblit, horizOn=False, vertOn=False, **lineprops)

        self.horizAxes = horizOn
        self.vertAxes = vertOn

        if len(horizOn) > 0:
            self.horizOn = True
        if len(vertOn) > 0:
            self.vertOn = True
        self.axes = axes
        xmin, xmax = axes[-1].get_xlim()
        ymin, ymax = axes[0].get_ylim()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        self.vlines = [ax.axvline(xmid, visible=True, **lineprops) for ax in self.vertAxes]
        self.hlines = [ax.axhline(ymid, visible=True, **lineprops) for ax in self.horizAxes]
        
class cartography():
    def __init__(self,path,deadPixeltol = 2,aspect="auto",Linescan = True,autoclose=False,save=False):
        self.shift_is_held = False
        self.path = path
        self.deadPixeltol = deadPixeltol
        self.aspect = aspect
        self.Linescan = Linescan
        self.leftpressed = False
        if autoclose:
            plt.ioff()
        else:
            plt.ion()
        
        dirname,filename = os.path.split(self.path)
        filename, ext = os.path.splitext(filename)
#    try:
#        infos = os.path.basename(dirname).split("-")
#        T = '' if not [x for x in infos if ('K' in x)] else [x for x in infos if ('K' in x)][0]
#        Sample = infos[1]
#        wire = '' if not [x for x in infos if ('fil' in x.lower())] else [x for x in infos if ('fil' in x.lower())][0]
#        if not wire:
#            wire = os.path.basename(os.path.dirname(dirname))
#        wire = wire.lower().replace('fil','Wire')
#    except:
#        T=''
#        Sample = ''
#        wire = ''
        hyppath = self.path
        specpath = os.path.join(dirname,filename+'_X_axis.asc')
        filepath = os.path.join(dirname,filename+'_SEM image after carto.tif')
        start = time.time()
        data = pd.read_csv(hyppath,delimiter='\t',header=None).to_numpy()
        end = time.time()
        print(end-start)
        #data = np.loadtxt(hyppath)
        xlen = int(data[0,0])   #nbr de pts selon x
        ylen = int(data[1,0])   #nbr de pts selon y
        wavelenght = np.loadtxt(specpath)
        self.wavelenght = wavelenght[:2048] #bins du spectro
        xcoord = data[0,1:]
        ycoord = data[1,1:]
        CLdata = data[2:,1:] #tableau de xlen * ylen points (espace) et 2048 longueur d'onde CLdata[:,n] n = numero du spectr
        self.hypSpectrum = np.transpose(np.reshape(np.transpose(CLdata),(ylen,xlen ,len(self.wavelenght))), (0, 1, 2))
    
    
        #correct dead / wrong pixels
        self.hypSpectrum, self.hotpixels = correct_dead_pixel(self.hypSpectrum,tol=self.deadPixeltol)

    #reduce the spectrum if wanted
#    if len(EnergyRange)==2:
#        
#        EnergyRange = eV_To_nm/np.array(EnergyRange) # en nm
#        EnergyRange.sort()
#        wavelenght = ma.masked_outside(wavelenght,*EnergyRange)
#        hypmask = np.resize(wavelenght.mask,hypSpectrum.shape)
#        hypSpectrum = ma.masked_array(hypSpectrum,mask=hypmask)
##        lidx = np.argmin(np.abs(wavelenght-EnergyRange[0]))
##        ridx = np.argmin(np.abs(wavelenght-EnergyRange[1]))
##        hypmask = np.zeros(hypSpectrum.shape,dtype=bool)
##        hypmask[:,:,indmin:indmax] = True
##        specmask = np.zeros(wavelenght.shape,dtype=bool)
##        specmask[lidx:ridx] = True
#    else:
        self.wavelenght = ma.masked_array(self.wavelenght,mask=False)
        hypmask = np.resize(self.wavelenght.mask,self.hypSpectrum.shape)
        self.hypSpectrum = ma.masked_array(self.hypSpectrum,mask=hypmask)
        
    
        self.linescan = np.sum(self.hypSpectrum,axis=0)
#    if normalise:
#        linescan = linescan/np.max(linescan,axis=1,keepdims=True)
#    linescan -= linescan.min()
        xscale_CL,yscale_CL,acceleration,image = scaleSEMimage(filepath)
        if self.Linescan:
            self.fig,(self.ax,self.bx,self.cx)=plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [1,1, 3]})
        else:
            self.fig,(self.ax,self.bx,self.cx)=plt.subplots(3,1,sharex=True,sharey=True)
        self.fig.patch.set_alpha(0) #Transparency style
        self.fig.subplots_adjust(top=0.9,bottom=0.12,left=0.15,right=0.82,hspace=0.1,wspace=0.05)
        newX = np.linspace(xscale_CL[int(xcoord.min())],xscale_CL[int(xcoord.max())],len(xscale_CL))
        newY = np.linspace(yscale_CL[int(ycoord.min())],yscale_CL[int(ycoord.max())],len(yscale_CL))
        self.X = np.linspace(np.min(newX),np.max(newX),self.hypSpectrum.shape[1])
        self.Y = np.linspace(np.min(newY),np.max(newY),self.hypSpectrum.shape[0])
    
        nImage = np.array(image.crop((xcoord.min(),ycoord.min(),xcoord.max(),ycoord.max())))
        self.ax.imshow(nImage,cmap='gray',vmin=0,vmax=65535,interpolation = "None",extent=[np.min(newX),np.max(newX),np.max(newY),np.min(newY)])
#        self.hypSpectrum = hypSpectrum-threshold
#        hypSpectrum = hypSpectrum*(hypSpectrum>=0)
        self.hypimage=np.sum(self.hypSpectrum,axis=2)
        #self.hypimage -= self.hypimage.min()
#    if log:
#        hypimage=np.log10(hypimage+1)
#        linescan = np.log10(linescan+1)
        #norm = matplotlib.colors.LogNorm(vmin=self.hypimage.min(), vmax=self.hypimage.max())
        self.lumimage = self.bx.imshow(self.hypimage,cmap='jet',interpolation = "None",extent=[np.min(newX),np.max(newX),np.max(newY),np.min(newY)])
#        self.lumimage = self.bx.pcolormesh(self.X,self.Y,self.hypimage,cmap='jet')
        if self.Linescan:
            
            self.im=self.cx.pcolormesh(self.X,eV_To_nm/self.wavelenght,self.linescan.T,cmap='jet',rasterized=True)
            def format_coord(x, y):
                xarr = self.X
                yarr = eV_To_nm/self.wavelenght
                if ((x > xarr.min()) and (x <= xarr.max()) and 
                    (y > yarr.min()) and (y <= yarr.max())):
                    col = np.argmin(abs(xarr-x))#np.searchsorted(xarr, x)-1
                    row = np.argmin(abs(yarr-y))#np.searchsorted(yarr, y)-1
                    z = self.linescan.T[row, col]
                    return f'x={x:1.4f}, y={y:1.4f}, lambda={eV_To_nm/y:1.2f}, z={z:1.2e}   [{row},{col}]'
                else:
                    return f'x={x:1.4f}, y={y:1.4f}, lambda={eV_To_nm/y:1.2f}'
            self.cx.format_coord = format_coord
            self.cx.set_ylabel("Energy (eV)")
            self.cx.set_xlabel("distance (µm)")
            self.cx.set_aspect('auto')
        else:
            self.im = self.cx.imshow(self.wavelenght[np.argmax(self.hypSpectrum,axis=2)],cmap='viridis',extent=[np.min(newX),np.max(newX),np.max(newY),np.min(newY)])    
            self.cx.set_aspect(aspect)
        self.bx.set_aspect(self.aspect)
        self.ax.set_aspect(self.aspect)
        self.ax.get_shared_y_axes().join(self.ax, self.bx)
        self.ax.set_ylabel("distance (µm)")
#    fig.text(ax.get_position().bounds[0]-0.11, ax.get_position().bounds[1],'distance (µm)',fontsize=16, va='center', rotation='vertical')
    
        pos = self.cx.get_position().bounds
        cbar_ax = self.fig.add_axes([0.85, pos[1], 0.05, pos[-1]*0.9])  
        self.fig.colorbar(self.im, cax=cbar_ax)
        cbar_ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    
        pos = self.bx.get_position().bounds
        cbar_ax = self.fig.add_axes([0.85, pos[1], 0.05, pos[-1]*0.9])
        self.fig.colorbar(self.lumimage,cax=cbar_ax)
        cbar_ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
        if save==True:
            self.fig.savefig(os.path.join(dirname,filename+".png"),dpi=300)
            #self.fig.savefig(os.path.join(dirname,filename+".svg"))
            
        if autoclose==True:
            plt.close(self.fig)
            return None
        else:
            self.spec_fig, self.spec_ax = plt.subplots()
            self.spec_data, = self.spec_ax.plot(eV_To_nm/self.wavelenght,np.zeros(self.wavelenght.shape[0]))
            self.spec_ax.set_ylim((0,self.hypSpectrum.max()))
            if Linescan:    
                self.cursor = MultiCursor(self.fig.canvas, (self.ax, self.bx), color='r', lw=1,horizOn=True, vertOn=True)
            def onmotion(event):
                if self.leftpressed:
                    x = event.xdata
                    y = event.ydata
                    if ((event.inaxes is not None) and (x > self.X.min()) and (x <= self.X.max()) and 
                        (y > self.Y.min()) and (y <= self.Y.max())):
                        indx = np.argmin(abs(x-self.X))
                        indy=np.argmin(abs(y-self.Y))
                        self.spec_data.set_ydata(self.hypSpectrum.data[indy,indx])
                        self.spec_fig.canvas.draw()
            
            def onclick(event):
                if event.button==1:
                    self.leftpressed = True
                    self.cursor.active = True
#                    if event.dblclick:
#                        self.cursor.active = not(self.cursor.active)
#                    elif self.cursor.active:
#                    x = event.xdata
#                    y = event.ydata
#                    if ((event.inaxes is not None) and (x > self.X.min()) and (x <= self.X.max()) and 
#                        (y > self.Y.min()) and (y <= self.Y.max())):
#                        indx = np.argmin(abs(x-self.X))
#                        indy=np.argmin(abs(y-self.Y))
#                        self.spec_data.set_ydata(self.hypSpectrum.data[indy,indx])
#                        self.spec_fig.canvas.draw_idle()
                elif event.button == 3:
                    self.wavelenght = ma.masked_array(self.wavelenght.data,mask=False)
                    hypmask = np.resize(self.wavelenght.mask,self.hypSpectrum.shape)
                    self.hypSpectrum = ma.masked_array(self.hypSpectrum.data,mask=hypmask)
                    self.hypimage=np.sum(self.hypSpectrum,axis=2)
                    #self.hypimage -= self.hypimage.min()
                    self.lumimage.set_array(self.hypimage)
                    self.cx.set_ylim(eV_To_nm/self.wavelenght.max(), eV_To_nm/self.wavelenght.min())
                    self.fig.canvas.draw_idle()
            def onrelease(event):
                if event.button==1:
                    self.leftpressed = False
                    self.cursor.active = False
            def onselect(ymin, ymax):
                indmin = np.argmin(abs(eV_To_nm/self.wavelenght-ymin))
                indmax = np.argmin(abs(eV_To_nm/self.wavelenght-ymax))
    #        
    ##            thisx = x[indmin:indmax]
                if abs(indmax-indmin)<1:return
    #            indmin, indmax = np.sort((indmax,indmin))
    #            thiswave = wavelenght[indmin:indmax]
    #            mask = np.zeros(hypSpectrum.shape)
    #            mask[:,:,indmin:indmax] = 1
    #            hypimage=np.sum(hypSpectrum*mask,axis=2)
    #            hypimage -= hypimage.min()
    #            lumimage.set_array(hypimage)
                self.wavelenght = ma.masked_outside(self.wavelenght,eV_To_nm/ymin,eV_To_nm/ymax)
                hypmask = np.resize(self.wavelenght.mask,self.hypSpectrum.shape)
                self.hypSpectrum = ma.masked_array(self.hypSpectrum,mask=hypmask)
                self.hypimage=np.sum(self.hypSpectrum,axis=2)
                #self.hypimage -= self.hypimage.min()
                self.lumimage.set_array(self.hypimage)
                self.cx.set_ylim(eV_To_nm/self.wavelenght.max(), eV_To_nm/self.wavelenght.min())
                self.fig.canvas.draw_idle()
            self.span = None
            if Linescan:
                self.span = SpanSelector(self.cx, onselect, 'vertical', useblit=True,
                                    rectprops=dict(alpha=0.5, facecolor='red'),button=1)
            self.fig.canvas.mpl_connect('button_press_event', onclick)
            self.fig.canvas.mpl_connect('motion_notify_event',onmotion)
            self.fig.canvas.mpl_connect('button_release_event',onrelease)
#            return self.hypSpectrum, self.wavelenght, spec_fig, spec_ax, cursor, span
            plt.show(block=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,help="path of the carto file")
    parser.add_argument("-dp", "--deadPixeltol", type=float, help="Number of sigma for dead pixel detection",default=2)
    parser.add_argument("-l", "--linescan", action="store_true",
                    help="Linescan or energycarto")
    parser.add_argument("-s", "--save", action="store_true",
                    help="save")
    args = parser.parse_args()

    cartography(path=args.path,deadPixeltol=args.deadPixeltol,Linescan=args.linescan,save=args.save)

#    path = r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-05-15 - T2455 Al - 300K\HYP1-T2455-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t005ms-cw430nm\Hyp.dat"
#    cartography(path=path,deadPixeltol=2,Linescan=True)

   