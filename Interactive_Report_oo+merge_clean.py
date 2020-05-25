# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:46:34 2019

@author: Sylvain
"""


import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(funcName)s:%(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os.path
import argparse
import matplotlib.cm as cm
#import pandas as pd
# import time
from Spectrum import Spectrum
from PIL.PngImagePlugin import PngImageFile, PngInfo
import scipy.constants as cst
eV_To_nm = cst.c*cst.h/cst.e*1E9
from hyperProcessing import correct_dead_pixel,loadData
#from SEMimage import scaleSEMimage
#from FileHelper import getListOfFiles,getSpectro_info
from matplotlib.widgets import MultiCursor, Cursor, SpanSelector
# plt.style.use('Rapport')
# plt.rc('axes', labelsize=16)
plt.rc('axes', labelsize=16,labelweight="medium")
plt.rc("figure", dpi=100)
plt.rc('xtick',labelsize = 14)
plt.rc('ytick',labelsize = 14)
plt.rc('svg',fonttype= 'none')

class MyMultiCursor(MultiCursor):
    def __init__(self, canvas, axes, useblit=True, horizOn=[], vertOn=[], **lineprops):
        super(MyMultiCursor, self).__init__(canvas, axes, useblit=useblit, horizOn=False, vertOn=False, **lineprops)

        self.horizAxes = horizOn
        self.vertAxes = vertOn

        if len(horizOn) > 0:
            self.horizOn = True
        if len(vertOn) > 0:
            self.vertOn = True

        xmin, xmax = axes[-1].get_xlim()
        ymin, ymax = axes[0].get_ylim()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        self.vlines = [ax.axvline(xmid, visible=True, **lineprops) for ax in self.vertAxes]
        self.hlines = [ax.axhline(ymid, visible=True, **lineprops) for ax in self.horizAxes]
        
class cartography(object):
    
    
    Emin = None
    Emax = None
    deadPixeltol = 2
    aspect = "auto"
    Type = "linescan"
    autoclose = False
    save = False
    lognorm = False
    trueSEM = False
    normalize = False
    autoplot = True
    fig_kw = {"figsize":[6.4,4.8]}
    gridspec_kw={'height_ratios': [1,1,2],"top":0.9,"bottom":0.12,"left":0.15,"right":0.82,"hspace":0.1,"wspace":0.05}
    subplot_kw=None
    def __init__(self,path,**kwargs):#deadPixeltol = 2,aspect="auto",Linescan = True,autoclose=False,save=False,lognorm = False,trueSEM=True,Emin=None,Emax=None,normalize=False):
        #%% Init
        self.shift_is_held = False
        self.leftpressed = False
        if type(path)==str:
            self.path = [path]
        else:
            self.path=path
        for key in kwargs:
            value_or_default = kwargs.get(key)
            setattr(self, key, value_or_default)
        # self.deadPixeltol = deadPixeltol
        # self.aspect = aspect
        # self.Linescan = Linescan
        # self.normalize = normalize
        if self.autoplot:
            self.load()
            self.plotLinescan(self.subplot_kw,self.gridspec_kw,**self.fig_kw)
            
    def load(self):            
        r = loadData(self.path,self.deadPixeltol)
        self.X = r['X']
        self.Y = r['Y']
        self.wavelenght = r['wavelenght']
        self.hypSpectrum = r['hyper']
        self.semImage = r["semImage"]
        if (self.Emin is not None) or (self.Emax is not None):
            self.Emin = np.max(eV_To_nm/self.wavelenght.min(),self.Emin)
            self.Emax = np.min(eV_To_nm/self.wavelenght.max(),self.Emax)
            #self.update(self.Emin,self.Emax)
        # self.hypSpectrum = np.roll(self.hypSpectrum,,axis=1)
        #self.semImage = np.roll(self.semImage,-12,axis=1)
        
        if self.trueSEM and r['semCarto'] is not None:
            self.semCarto = r['semCarto']
        else:
            self.trueSEM = False
    def plotLinescan(self,subplot_kw=None, gridspec_kw=None, **fig_kw):
        if self.autoclose:
            plt.ioff()
        else:
            plt.ion()
        #%% Plot
        self.fig,(self.ax,self.bx,self.cx)=plt.subplots(3,1,sharex=True,subplot_kw=subplot_kw,gridspec_kw=gridspec_kw,**fig_kw)
        self.fig.patch.set_alpha(0) #Transparency style
        self.fig.subplots_adjust(top=0.9,bottom=0.12,left=0.15,right=0.82,hspace=0.1,wspace=0.05)

        if (self.trueSEM):
            logger.info("True SEM = %s"%str(self.trueSEM))
            self.ax.imshow(self.semCarto,cmap='gray',interpolation = 'nearest',\
                           extent=[np.min(self.X),np.max(self.X),np.max(self.Y),np.min(self.Y)])
        else:
            self.ax.imshow(self.semImage,cmap='gray',vmin=0,vmax=65535,\
                           interpolation = 'nearest',\
                               extent=[np.min(self.X),np.max(self.X),np.max(self.Y),np.min(self.Y)])
        self.hypimage=np.nansum(self.hypSpectrum,axis=2)
        jet = cm.jet
        jet.set_bad(color='k')
        self.hyperspectralmap = cm.ScalarMappable(cmap=jet)
        self.hyperspectralmap.set_clim(vmin=np.nanmin(self.hypimage),vmax=np.nanmax(self.hypimage))
        
        self.lumimage = self.bx.imshow(self.hypimage,cmap=self.hyperspectralmap.cmap,\
                                           norm=self.hyperspectralmap.norm,\
                                           interpolation = 'nearest',\
                                           extent=[np.min(self.X),np.max(self.X),np.max(self.Y),np.min(self.Y)])
            
        self.linescan = np.sum(self.hypSpectrum,axis=0)
        if self.normalize:
            self.linescan /= self.linescan.max()
        if self.lognorm:
            norm = matplotlib.colors.LogNorm()#vmin=np.nanmin(self.linescan),vmax=np.nanmax(self.linescan))
        else:
            norm = matplotlib.colors.Normalize()#vmin=np.nanmin(self.linescan),vmax=np.nanmax(self.linescan))
        self.linescanmap = cm.ScalarMappable(cmap=jet,norm=norm)
        self.linescanmap.set_clim(vmin=np.nanmin(self.linescan),vmax=np.nanmax(self.linescan))
        # ATTENTION pcolormesh =/= imshow
        #imshow takes values at pixel 
        #pcolormesh takes values between
        temp = np.linspace(self.X.min(),self.X.max(),self.X.size+1)
        self.im=self.cx.pcolormesh(temp,eV_To_nm/self.wavelenght,self.linescan.T,\
                                   cmap=self.linescanmap.cmap,norm=self.linescanmap.norm,\
                                       rasterized=True,antialiased=True)
        
        self.cx.format_coord = self.format_coord
        self.cx.set_ylabel("Energy (eV)")
        self.cx.set_xlabel("Distance (µm)")
        self.cx.set_aspect(self.aspect)

        self.bx.set_aspect(self.aspect)
        self.ax.set_aspect(self.aspect)
        self.ax.get_shared_y_axes().join(self.ax, self.bx)
        self.ax.set_ylabel("Distance (µm)")
    
        #Colorbar du linescan
        pos = self.cx.get_position().bounds
        cbar_ax = self.fig.add_axes([0.85, pos[1], 0.05, pos[-1]*0.9])  
        self.fig.colorbar(self.linescanmap, cax=cbar_ax)
        if not self.lognorm:
            cbar_ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    
        #Colorbar de l'image
        pos = self.bx.get_position().bounds
        cbar_ax = self.fig.add_axes([0.85, pos[1], 0.05, pos[-1]*0.9])
        self.fig.colorbar(self.hyperspectralmap,cax=cbar_ax)
        cbar_ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
        #%%
        # if save==True:
        #     self.fig.savefig(os.path.join(dirname,filename+".png"),dpi=300)
        if self.autoclose==True:
            plt.close(self.fig)
            return None
        self.Spectrum = Spectrum(self.wavelenght,np.nanmax(self.hypSpectrum))
        self.Spectrum.plot_meanSpectrum(self.wavelenght, np.mean(self.hypSpectrum,axis=(0,1)))
        #if Linescan:    
            #self.cursor = MultiCursor(self.fig.canvas, (self.ax, self.bx), color='r', lw=1,horizOn=True, vertOn=True)
        
        # def on_xlims_change(axes):
        #     print("updated xlims: ", axes)
        
        self.span = None
        self.span = SpanSelector(self.cx, self.onselect, 'vertical', useblit=True,
                            rectprops=dict(alpha=0.5, facecolor='red'),button=1)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event',self.onmotion)
        self.fig.canvas.mpl_connect('button_release_event',self.onrelease)
#            return self.hypSpectrum, self.wavelenght, spec_fig, spec_ax, cursor, span
        plt.show()
        #plt.show(block=True)
    
    
    def add_meta_data(self,meta):
        # use monkey-patching to replace the original plt.figure() function with
        # our own, which supports clipboard-copying
        mpl_savefig = plt.Figure.savefig
        
        def savemeta(*args, **kwargs):
            path = args[1]
            if path.endswith(".fig"):
                import pickle as pkl
                with open(path, "wb") as file:
                    pkl.dump(self.fig, file)
            else:
                mpl_savefig(*args,**kwargs)
                #fig = args[0]
                if path.endswith(".png"):
                    targetImage = PngImageFile(path)
                    metadata = PngInfo()
                    metadata.add_text("Description", str(meta))
                    targetImage.save(path, pnginfo=metadata)
            
        plt.Figure.savefig = savemeta
    
    def format_coord(self,x, y):
        xarr = self.X
        yarr = eV_To_nm/self.wavelenght
        if ((x > xarr.min()) and (x <= xarr.max()) and 
            (y > yarr.min()) and (y <= yarr.max())):
            col = np.searchsorted(xarr, x)-1#np.argmax(abs(xarr-x))#
            row = np.argmin(abs(yarr-y))#np.searchsorted(yarr, y)-1#
            z = self.linescan.T[row, col]
            return f'x={x:1.4f}, y={y:1.4f}, lambda={eV_To_nm/y:1.2f}, z={z:1.2e}   [{row},{col}]'
        else:
            return f'x={x:1.4f}, y={y:1.4f}, lambda={eV_To_nm/y:1.2f}'
    
    def onmotion(self,event):
        if self.leftpressed:
            x = event.xdata
            y = event.ydata
            if ((event.inaxes is not None) and (x > self.X.min()) and (x <= self.X.max()) and 
                (y > self.Y.min()) and (y <= self.Y.max())):
                indx = np.argmin(abs(x-self.X))
                indy=np.argmin(abs(y-self.Y))
                self.Spectrum.set_y(self.hypSpectrum.data[indy,indx])
                self.Spectrum.update()
    def onclick(self,event):
        if event.button==1:
            self.leftpressed = True
            self.onmotion(event)
        elif event.button == 3:
            self.update()
            #self.cursor.active = False   
    def onselect(self,ymin, ymax):
        indmin = np.argmin(abs(eV_To_nm/self.wavelenght-ymin))
        indmax = np.argmin(abs(eV_To_nm/self.wavelenght-ymax))
        #indmin resp.max est lindex tel que wavelenght[indmin] minimise
        # la distance entre la position du clic et la longueur d'onde
#                print(eV_To_nm/self.wavelenght[indmin])
#                print(eV_To_nm/self.wavelenght[indmax])
        if abs(indmax-indmin)<1:return
        self.update(ymin,ymax)
    def onrelease(self,event):
        if event.button==1:
            self.leftpressed = False
    def update(self,ymin=None,ymax=None):
        if ((ymin is None) & (ymax is None)):
            self.wavelenght = ma.masked_array(self.wavelenght.data,mask=False)
        else:
            self.wavelenght = ma.masked_outside(self.wavelenght.data,eV_To_nm/ymax,eV_To_nm/ymin)
        self.Emin = eV_To_nm/self.wavelenght.max()
        self.Emax = eV_To_nm/self.wavelenght.min()
        self.Spectrum.set_limitbar(self.wavelenght.min(),self.wavelenght.max())
        hypmask = np.resize(self.wavelenght.mask,self.hypSpectrum.shape)
        self.hypSpectrum = ma.masked_array(self.hypSpectrum.data,mask=hypmask)
        self.hypimage=np.nansum(self.hypSpectrum,axis=2)
        self.lumimage.set_array(self.hypimage)        
        self.linescan = np.sum(self.hypSpectrum,axis=0)
        if self.normalize:
            self.linescan /= self.linescan.max()
        #self.im.set_array((self.linescan.T[:,:]).ravel())#flat shading
        #self.im.set_array((self.linescan.T).ravel())#gouraud shading
        
        self.linescanmap.set_clim(vmin=np.nanmin(self.linescan),vmax=np.nanmax(self.linescan))
        self.hyperspectralmap.set_clim(vmin=np.nanmin(self.hypimage),vmax=np.nanmax(self.hypimage))
        self.cx.set_ylim(eV_To_nm/self.wavelenght.max(), eV_To_nm/self.wavelenght.min())
        self.im.set_norm(self.linescanmap.norm)
        self.fig.canvas.draw_idle()
        self.Spectrum.update()
        #self.fig.canvas.blit(self.fig.bbox) #necessaire d'utiliser blit si ajout de curseur

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=str,nargs='*',help="paths of the carto file")
    parser.add_argument("-dp", "--deadPixeltol", type=float, help="Number of sigma for dead pixel detection",default=2)
    parser.add_argument("-l", "--linescan", action="store_true",
                    help="Linescan or energycarto")
    parser.add_argument("-s", "--save", action="store_true",
                    help="save")
    args = parser.parse_args()
    print(args.paths)
    matplotlib.rcParams["savefig.directory"] = ""
    
    autoplot = not bool(input("Edit Default Param. [y/n]?").lower()=='y')
    # log = bool(input("Logscale [y/n]?").lower()=='y')
    # trueSEM = bool(input("True SEM [y/n]?").lower()=='y')
    # normalize = bool(input("Normalize [y/n]?").lower()=='y')
    
    
    carto = cartography(path=args.paths,deadPixeltol=args.deadPixeltol,Linescan=args.linescan,autoplot=autoplot)#lognorm=log,trueSEM=trueSEM,normalize=normalize)

    
    # path = [r"C:\Users\sylvain.finot\Cloud Neel\Data\2019\2019-04-29 - T2628 - 300K\Fil 1\HYP-T2628-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t025ms-cw550nm\Hyp.dat",
# r"C:\Users\sylvain.finot\Cloud Neel\Data\2019\2019-04-29 - T2628 - 300K\Fil 1\HYP-T2628-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t025ms-cw360nm\Hyp.dat",r"C:\Users\sylvain.finot\Cloud Neel\Data\2019\2019-04-29 - T2628 - 300K\Fil 1\HYP-T2628-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t025ms-cw450nm\Hyp.dat"]
#     path = r"C:\Users\sylvain.finot\Cloud Neel\Data\2019\2019-04-29 - T2628 - 300K\Fil 1\HYP-T2628-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t025ms-cw550nm\Hyp.dat"
#     #    if len(sys.argv)>1:
#     # if len(sys.argv)>1:
#     #     path = sys.argv[1:]
# #        cartography(path=paths,deadPixeltol=2,Linescan=True)
# #    path = [r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-05-15 - T2455 Al - 300K\HYP1-T2455-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t005ms-cw430nm\Hyp.dat"]
#     # matplotlib.rcParams["savefig.directory"] = ""
#     # start = time.time() 
#     cl = cartography(path=path,deadPixeltol=2,Linescan=True,Emin=2.25)
#     hyp = cl.hypSpectrum
    # end = time.time()
    #print(end-start)

   