# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:58:12 2019

@author: sylvain.finot
"""


import scipy.constants as cst
eV_To_nm = cst.c*cst.h/cst.e*1E9
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
import os.path
import sys
from scipy.signal import savgol_filter
plt.style.use('Rapport')
#plt.rc('axes', labelsize=16)


def plotSpectrum(paths,eV,savgol):
    if type(paths)==str:
        paths = [paths]
    fig=plt.figure()
    fig.patch.set_alpha(0)
    ax1=SubplotHost(fig, 111)
    fig.add_subplot(ax1)
    for p in paths:     
        data = np.loadtxt(p,skiprows=9)
        if max(data[:,0])>100:
            isIneV=False
        else:
            isIneV=True
        print("eV :",eV)
        print("isIneV",isIneV)
        if savgol:
            data[:,1] = savgol_filter(data[:,1],101,2)
        if eV ^ isIneV:
            ax1.plot(eV_To_nm/data[:,0],data[:,1])
        else:
            ax1.plot(data[:,0],data[:,1])
    
    ax1.set_ylabel('Intensity (arb. units)')
    ax2=ax1.twin()
    if eV:
        ax1.set_xlabel('Energy (eV)')
        ax2.set_xlabel('Wavelength (nm)')
        tticks=np.array(np.round(eV_To_nm/ax1.get_xticks(),0),np.int)
    else:
        ax2.set_xlabel('Energy (eV)')
        ax1.set_xlabel('Wavelength (nm)')
        tticks=np.round(eV_To_nm/ax1.get_xticks(),2)
    
    def on_xlims_change(axes):
        print("updated xlims: ", ax1.get_xlim())
        if eV:
            tticks=np.array(np.round(eV_To_nm/ax1.get_xticks(),0),np.int)
        else:
            tticks=np.round(eV_To_nm/ax1.get_xticks(),2)
        ax2.set_xticks( [ eV_To_nm/t for t in tticks ] )
        ax2.set_xticklabels(tticks)
        
    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    ax2.set_xticks( [ eV_To_nm/t for t in tticks ] )
    ax2.set_xticklabels(tticks)
    #ax2.axis["top"].label.set_visible(True)
    ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
    ax2.set_yticks([])
    fig.subplots_adjust(top=0.885, bottom=0.125, left=0.13, right=0.94,)
    plt.show(block=True)



#def main():
#    path = input("Enter the path of your file: ")
#    path=path.replace('"','')
#    path=path.replace("'",'')
##    path = r'C:/Users/sylvain.finot/Documents/data/2019-03-11 - T2597 - 5K/Fil3/TRCL-cw455nm/TRCL.dat'
#    plotSpectrum(path)
#    
if __name__ == '__main__':
    if len(sys.argv)>1:
        paths = sys.argv[1:]
        eV = input("Main axis in eV? [y/n] : ")
        eV = "y" in eV
        savgol = input("filter? [y/n] : ")
        savgol = "y" in savgol
        plotSpectrum(paths,eV,savgol)
        #input("Press ENTER to exit")