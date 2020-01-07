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
import matplotlib.cm as cm
import pandas as pd
import time
import scipy.constants as cst
eV_To_nm = cst.c*cst.h/cst.e*1E9
from detect_dead_pixels import correct_dead_pixel
from SEMimage import scaleSEMimage
from FileHelper import getListOfFiles,getSpectro_info
from matplotlib.widgets import MultiCursor, Cursor, SpanSelector
plt.style.use('Rapport')
plt.rc('axes', labelsize=16)
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(module)s:%(levelname)s:%(message)s',datefmt='%H:%M:%S')

def compareManyArrays(arrayList):
    return (np.diff(np.vstack(arrayList).reshape(len(arrayList),-1),axis=0)==0).all()                
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
        
class cartography():
    def __init__(self,path,deadPixeltol = 2,aspect="auto",Linescan = True,autoclose=False):
        self.shift_is_held = False
        if type(path)==str:
            self.path = [path]
        else:
            self.path=path
        self.deadPixeltol = deadPixeltol
        self.aspect = aspect
        self.Linescan = Linescan
        self.leftpressed = False
        
        if autoclose:
            plt.ioff()
        else:
            plt.ion()
        #############################################################
        dirnameL = list()
        wavelenghtL = list()
        infoL = list()
        CLdataL = list()
        xlenL = list()
        ylenL = list()
        xcoordL = list()
        ycoordL = list()
        filepathL = list()
        for i in range(len(path)):
            dirname,filename = os.path.split(path[i])
            filename, ext = os.path.splitext(filename)
            dirnameL.append(os.path.join(dirname,filename))
            hyppath = path[i]
            specpath = os.path.join(dirnameL[i]+'_X_axis.asc')
            filepathL.append(os.path.join(dirnameL[i]+'_SEM image after carto.tif'))
            #data = np.loadtxt(hyppath)
            data = pd.read_csv(hyppath,delimiter='\t',header=None,dtype=np.float).to_numpy()
            xlenL.append(int(data[0,0]))
            ylenL.append(int(data[1,0]))
            wavelenghtL.append(np.loadtxt(specpath)[:2048])
            xcoordL.append(data[0,1:])
            ycoordL.append(data[1,1:])
            CLdataL.append(data[2:,1:])
            spectroInfos = getSpectro_info(os.path.join(dirnameL[i]+'_spectro_info.asc'))
            logger.info(spectroInfos)
            infoL.append(spectroInfos[:,1])
            
        wavelenghtL = np.array(wavelenghtL)
        inds = wavelenghtL.argsort(axis=0)[:,0]
        CLdataL = list(np.array(CLdataL)[inds])
        xlen = np.array(xlenL)[inds]
        ylen = np.array(ylenL)[inds]
        xcoord = np.array(xcoordL)[inds]
        ycoord = np.array(ycoordL)[inds]
        wavelenght = list(wavelenghtL[inds])
        
        SameConditions = compareManyArrays(infoL) & compareManyArrays(xcoordL) & compareManyArrays(xcoordL)
        print('Same Conditions : ', SameConditions)
        
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
            
        self.wavelenght = np.array(wavelenght)[0]
        CLdata = np.array(CLdataL)[0]
        xlen = np.array(xlenL)[0]
        ylen = np.array(ylenL)[0]
        xcoord = np.array(xcoordL)[0]
        ycoord = np.array(ycoordL)[0]
        filepath = filepathL[0]
        #############################################################
#        dirname,filename = os.path.split(self.path)
#        filename, ext = os.path.splitext(filename)
#
#        hyppath = self.path
#        specpath = os.path.join(dirname,filename+'_X_axis.asc')
#        filepath = os.path.join(dirname,filename+'_SEM image after carto.tif')
#        data = np.loadtxt(hyppath)
#        xlen = int(data[0,0])   #nbr de pts selon x
#        ylen = int(data[1,0])   #nbr de pts selon y
#        wavelenght = np.loadtxt(specpath)
#        self.wavelenght = wavelenght[:2048] #bins du spectro
#        xcoord = data[0,1:]
#        ycoord = data[1,1:]
#        CLdata = data[2:,1:] #tableau de xlen * ylen points (espace) et 2048 longueur d'onde CLdata[:,n] n = numero du spectr
#        
#        
        
        self.hypSpectrum = np.transpose(np.reshape(np.transpose(CLdata),(ylen,xlen ,len(self.wavelenght))), (0, 1, 2))
        #correct dead / wrong pixels
        #if len(self.path)==1:
        #    self.hypSpectrum, self.hotpixels = correct_dead_pixel(self.hypSpectrum,tol=self.deadPixeltol)

        self.wavelenght = ma.masked_array(self.wavelenght,mask=False)
        hypmask = np.resize(self.wavelenght.mask,self.hypSpectrum.shape)
        self.hypSpectrum = ma.masked_array(self.hypSpectrum,mask=hypmask)
        
    
        
        #self.linescan -= np.nanmin(self.linescan[np.nan_to_num(self.linescan)>0])
        #self.linescan[self.linescan<0] = 0 #empeche d'avoir des valeurs négatives la où il manque des données
#    if normalise:
#        linescan = linescan/np.max(linescan,axis=1,keepdims=True)
#    linescan -= linescan.min()
        xscale_CL,yscale_CL,acceleration,image = scaleSEMimage(filepath)
        if self.Linescan:
            self.fig,(self.ax,self.bx,self.cx)=plt.subplots(3,1,sharex=True,gridspec_kw={'height_ratios': [1,1,3]})
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
        self.hypimage=np.nansum(self.hypSpectrum,axis=2)
        jet = cm.get_cmap("jet")
        jet.set_bad(color='k')
        self.hyperspectralmap = cm.ScalarMappable(cmap=jet)
        self.hyperspectralmap.set_clim(vmin=np.nanmin(self.hypimage),vmax=np.nanmax(self.hypimage))
        #self.hypimage -= np.nanmin(self.hypimage)
#    if log:
#        hypimage=np.log10(hypimage+1)
#        linescan = np.log10(linescan+1)
        self.lumimage = self.bx.imshow(self.hypimage,cmap=self.hyperspectralmap.cmap,norm=self.hyperspectralmap.norm,interpolation = "None",extent=[np.min(newX),np.max(newX),np.max(newY),np.min(newY)])
#        self.lumimage = self.bx.pcolormesh(self.X,self.Y,self.hypimage,cmap='jet')
        if self.Linescan:
            self.linescan = np.nansum(self.hypSpectrum,axis=0)
            self.linescanmap = cm.ScalarMappable(cmap=jet)
            self.linescanmap.set_clim(vmin=np.nanmin(self.linescan),vmax=np.nanmax(self.linescan))
            temp = np.linspace(newX.min(),newX.max(),self.X.size+1)
            self.im=self.cx.pcolormesh(temp,eV_To_nm/self.wavelenght,self.linescan.T,cmap=self.linescanmap.cmap,rasterized=True)
            def format_coord(x, y):
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
            self.cx.format_coord = format_coord
            self.cx.set_ylabel("Energy (eV)")
            self.cx.set_xlabel("distance (µm)")
            self.cx.set_aspect('auto')
        else:
            self.im = self.cx.imshow(self.wavelenght[np.nanargmax(self.hypSpectrum,axis=2)],cmap='viridis',extent=[np.min(newX),np.max(newX),np.max(newY),np.min(newY)])    
            self.cx.set_aspect(aspect)
        self.bx.set_aspect(self.aspect)
        self.ax.set_aspect(self.aspect)
        self.ax.get_shared_y_axes().join(self.ax, self.bx)
        self.ax.set_ylabel("distance (µm)")
    
        #Colorbar du linescan
        pos = self.cx.get_position().bounds
        cbar_ax = self.fig.add_axes([0.85, pos[1], 0.05, pos[-1]*0.9])  
        self.fig.colorbar(self.linescanmap, cax=cbar_ax)
        cbar_ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
    
        #Colorbar de l'image
        pos = self.bx.get_position().bounds
        cbar_ax = self.fig.add_axes([0.85, pos[1], 0.05, pos[-1]*0.9])
        self.fig.colorbar(self.hyperspectralmap,cax=cbar_ax)
        cbar_ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
#    if save==True:
#        savedir = os.path.join(dirname,'Saved')
#        if not os.path.exists(savedir):
#            os.mkdir(savedir)
#        savename = 'SEM+Hyp+Linescan'
#        if ((len(T)>0) & (len(Sample)>0) & (len(wire)>0)):
#            savename='%s_%s_%s_%s'%(savename,Sample,T,wire)
#        fig.savefig(os.path.join(dirname,'Saved',savename+".png"),dpi=300)
            
        if autoclose==True:
            plt.close(self.fig)
            return None
        else:
            self.spec_fig, self.spec_ax = plt.subplots()
            self.spec_data, = self.spec_ax.plot(eV_To_nm/self.wavelenght,np.zeros(self.wavelenght.shape[0]))
            self.spec_ax.set_ylim((0,np.nanmax(self.hypSpectrum)))
            self.spec_ax.set_xlabel('Energy (eV)')
            self.spec_ax.set_ylabel('Intensity (a.u)')
            self.spec_fig.subplots_adjust(top=0.885, bottom=0.125, left=0.13, right=0.94)
            self.spec_ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            self.spec_minbar = self.spec_ax.axvline(eV_To_nm/self.wavelenght.max())
            self.spec_maxbar = self.spec_ax.axvline(eV_To_nm/self.wavelenght.min())
            #if Linescan:    
                #self.cursor = MultiCursor(self.fig.canvas, (self.ax, self.bx), color='r', lw=1,horizOn=True, vertOn=True)
            def onmotion(event):
                if self.leftpressed:
                    x = event.xdata
                    y = event.ydata
                    if ((event.inaxes is not None) and (x > self.X.min()) and (x <= self.X.max()) and 
                        (y > self.Y.min()) and (y <= self.Y.max())):
                        indx = np.argmin(abs(x-self.X))
                        indy=np.argmin(abs(y-self.Y))
                        self.spec_data.set_ydata(self.hypSpectrum.data[indy,indx])
                        self.spec_fig.canvas.draw_idle()
            
            def onclick(event):
                if event.button==1:
                    self.leftpressed = True
                    #self.cursor.active = True
                    #if event.dblclick:
#                        if not(self.cursor.visible):
#                            self.fig.canvas.blit(self.fig.bbox)
                        #self.cursor.active = not(self.cursor.active)
#                        self.cursor.visible = not(self.cursor.visible)
#                        self.fig.canvas.draw_idle()
#                        self.fig.canvas.blit(self.fig.bbox)
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
                    self.hypimage=np.nansum(self.hypSpectrum,axis=2)
                    self.lumimage.set_array(self.hypimage)
                    self.hyperspectralmap.set_clim(vmin=np.nanmin(self.hypimage),vmax=np.nanmax(self.hypimage))
                    self.cx.set_ylim(eV_To_nm/self.wavelenght.max(), eV_To_nm/self.wavelenght.min())
                    # TODO : A recoder
                    self.linescan = np.nansum(self.hypSpectrum,axis=0)
                    self.spec_maxbar.set_xdata(np.repeat(eV_To_nm/self.wavelenght.min(),2))
                    self.spec_minbar.set_xdata(np.repeat(eV_To_nm/self.wavelenght.max(),2))
                    self.im.set_array((self.linescan.T[:,:]).ravel())#flat shading
                    #self.im.set_array((self.linescan.T).ravel())#gouraud shading
                    self.linescanmap.set_clim(vmin=np.nanmin(self.linescan),vmax=np.nanmax(self.linescan))
                    self.im.set_norm(self.linescanmap.norm)
                    self.im.set_cmap(self.linescanmap.cmap)
                    self.fig.canvas.draw_idle()
                    self.spec_fig.canvas.draw_idle()
                    self.fig.canvas.blit(self.fig.bbox)
            def onrelease(event):
                if event.button==1:
                    self.leftpressed = False
                    #self.cursor.active = False
            def onselect(ymin, ymax):
                indmin = np.argmin(abs(eV_To_nm/self.wavelenght-ymin))
                indmax = np.argmin(abs(eV_To_nm/self.wavelenght-ymax))
                #indmin resp.max est lindex tel que wavelenght[indmin] minimise
                # la distance entre la position du clic et la longueur d'onde
#                print(eV_To_nm/self.wavelenght[indmin])
#                print(eV_To_nm/self.wavelenght[indmax])
                if abs(indmax-indmin)<1:return

                self.wavelenght = ma.masked_outside(self.wavelenght.data,eV_To_nm/ymax,eV_To_nm/ymin)
                hypmask = np.resize(self.wavelenght.mask,self.hypSpectrum.shape)
                self.hypSpectrum = ma.masked_array(self.hypSpectrum.data,mask=hypmask)
                self.hypimage=np.nansum(self.hypSpectrum,axis=2)
                self.lumimage.set_array(self.hypimage)
                self.hyperspectralmap.set_clim(vmin=np.nanmin(self.hypimage),vmax=np.nanmax(self.hypimage))
                self.cx.set_ylim(eV_To_nm/self.wavelenght.max(), eV_To_nm/self.wavelenght.min())
                self.linescan = np.nansum(self.hypSpectrum,axis=0)
                self.spec_maxbar.set_xdata(np.repeat(eV_To_nm/self.wavelenght.min(),2))
                self.spec_minbar.set_xdata(np.repeat(eV_To_nm/self.wavelenght.max(),2))
                self.im.set_array((self.linescan.T[:,:]).ravel())#flat shading
                #self.im.set_array((self.linescan.T).ravel())#gouraud shading
                self.linescanmap.set_clim(vmin=np.nanmin(self.linescan),vmax=np.nanmax(self.linescan))
                self.im.set_norm(self.linescanmap.norm)
                self.im.set_cmap(self.linescanmap.cmap)
                self.fig.canvas.draw_idle()
                self.spec_fig.canvas.draw_idle()
                self.fig.canvas.blit(self.fig.bbox)
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
    import sys
#    parser = argparse.ArgumentParser()
#    parser.add_argument("path", type=list,help="path of the carto file")
#    parser.add_argument("-dp", "--deadPixeltol", type=float, help="Number of sigma for dead pixel detection",default=2)
#    parser.add_argument("-l", "--linescan", action="store_true",
#                    help="Linescan or energycarto")
#    parser.add_argument("-s", "--save", action="store_true",
#                    help="save")
#    args = parser.parse_args()
#    print(args.path)
    #cartography(path=args.path,deadPixeltol=args.deadPixeltol,Linescan=args.linescan)
    path = [r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-04-29 - T2628 - 300K\Fil 1\HYP-T2628-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t025ms-cw550nm\Hyp.dat",
r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-04-29 - T2628 - 300K\Fil 1\HYP-T2628-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t025ms-cw360nm\Hyp.dat"]#r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-04-29 - T2628 - 300K\Fil 1\HYP-T2628-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t025ms-cw450nm\Hyp.dat"]
#    if len(sys.argv)>1:
    if len(sys.argv)>1:
        path = sys.argv[1:]
#        cartography(path=paths,deadPixeltol=2,Linescan=True)
#    path = [r"C:\Users\sylvain.finot\Cloud Neel\Data\2019-05-15 - T2455 Al - 300K\HYP1-T2455-300K-Vacc5kV-spot5-zoom8000x-gr600-slit0-2-t005ms-cw430nm\Hyp.dat"]
    matplotlib.rcParams["savefig.directory"] = ""
    start = time.time() 
    cartography(path=path,deadPixeltol=2,Linescan=True)
    end = time.time()
    print(end-start)

   