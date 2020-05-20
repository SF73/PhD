# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:10:17 2020

@author: Sylvain.FINOT
"""

import numpy as np
import matplotlib.pyplot as plt
from ReadPhu import readphu
import sys
import logging
import itertools
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(module)s:%(levelname)s:%(message)s',datefmt='%H:%M:%S')
plt.style.use("Rapport")


def normalisation(t,fclock,fcount,T,dt,delay=0):
    return fcount*fclock*T*dt * np.exp(-fclock*np.abs(t-delay))

def normalisation2(t,I,G,eff,BS,T,dt,delay=0):
    """
    

    Parameters
    ----------
    t : float
        bins of the histogram in ns.
    I : float
        Ebeam current in pA.
    G : int
        Number of photon generated per electron.
    eff : float
        efficiency.
    BS : float
        percentage [0,1] of the signal going on the detector.
    T : float
        Integration time in second.
    dt : int
        Resolution of the hist in ps
    delay : float, optional
        Delay between the 2 detectors?. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    eRate = (I/1.602e-19)*1e-21
    N = eRate*G*eff
    return normalisation(t,N*BS*1e9,N*(1-BS)*1e9,T,dt,delay)
    #return N**2*(T*1e9)*(dt*1e-3)*BS*(1-BS),np.exp(-N*(1-BS)*np.abs(t-delay))

def model(t,t0,tau,g2):
 return 1+g2*np.exp(-np.abs((t-t0))/tau)

def fit(t,data):
 tmax = t[np.argmax(data)]
 mean = np.mean(data[(t>60) & (t<80)])
 ndata = data/mean
 fdata = savgol_filter(ndata,11,2)
 tau = abs(tmax-t[np.argmin(np.abs(fdata-fdata.max()*np.exp(-1)))])
 p0 = [tmax,tau,max(ndata)]
 idx = np.where(t<80)
 popt,pcov = curve_fit(model,t[idx],ndata[idx],p0)
 return mean,popt



if __name__ == '__main__':
    if len(sys.argv)>1:
        fig, ax =plt.subplots()
        paths = sys.argv[1:]
        for p in paths:
            # inputfile = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-01-10 - HBT\Mesure_g2_T300K_PassHaut400nm_2455_spot3_HV3kV_zoom1600000_binning512ps.phu"
            # t,counts,tags = readphu(inputfile) 
            t,counts,tags = readphu(p)
            tags = dict(tags)
            SyncDivider = tags['HistResDscr_HWSyncDivider(0)']
            SyncRate = tags['HistResDscr_SyncRate(0)']
            CountRate = tags['HistResDscr_InputRate(0)']
            HistRate = tags["HistResDscr_HistCountRate(0)"]
            BinningFactor = tags['HistResDscr_MDescBinningFactor(0)']
            Resolution = tags['HistResDscr_MDescResolution(0)']
            IntegrationTime = tags['HistResDscr_MDescStopAfter(0)']
            BS = CountRate/(CountRate+SyncRate)
            logger.info('\nSyncDivider : %s\nSyncRate : %s\nCountRate : %s\nBeamSplitter : %s\nBinningFactor : %s\nResolution : %.3E\nIntegrationTime : %s'%(SyncDivider,SyncRate,CountRate,BS,BinningFactor,Resolution,IntegrationTime))
            logger.info('HistCountRate : %s'%HistRate)
            color=next(ax._get_lines.prop_cycler)['color']
            mean,popt = fit(t*1e9,counts)
            ax.plot(t*1e9,counts/mean,'.',c=color,alpha=0.9)
        ax.set_xlabel("Delay (ns)")
        ax.set_ylabel("Counts")
        plt.show(block=True)