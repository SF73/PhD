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
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(module)s:%(levelname)s:%(message)s',datefmt='%H:%M:%S')
plt.style.use("Rapport")


def normalisation(t,fclock,fcount,T,dt,delay=0):
    return fcount*fclock*T*dt * np.exp(-fclock*np.abs(t-delay))

if __name__ == '__main__':
    if len(sys.argv)>1:
        fig, ax =plt.subplots()
        paths = sys.argv[1:]
        for p in paths:
#inputfile = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-01-10 - HBT\Mesure_g2_T300K_PassHaut400nm_2455_spot3_HV3kV_zoom1600000_binning512ps.phu"
            t,counts,tags = readphu(p)
            tags = dict(tags)
            SyncDivider = tags['HistResDscr_HWSyncDivider(0)']
            SyncRate = tags['HistResDscr_SyncRate(0)']
            CountRate = tags['HistResDscr_InputRate(0)']
            BinningFactor = tags['HistResDscr_MDescBinningFactor(0)']
            Resolution = tags['HistResDscr_MDescResolution(0)']
            IntegrationTime = tags['HistResDscr_MDescStopAfter(0)']
            BS = CountRate/(CountRate+SyncRate)
            logger.info('\nSyncDivider : %s\nSyncRate : %s\nCountRate : %s\nBeamSplitter : %s\nBinningFactor : %s\nResolution : %.3E\nIntegrationTime : %s'%(SyncDivider,SyncRate,CountRate,BS,BinningFactor,Resolution,IntegrationTime))
            color=next(ax._get_lines.prop_cycler)['color']
            ax.plot(t*1e9,counts,'.',c=color,alpha=0.1)
            #ax.plot(t*1e9,normalisation(t,SyncRate,CountRate,IntegrationTime*1e-3,Resolution),c=color)
        ax.set_xlabel("Delay (ns)")
        ax.set_ylabel("Counts")
        plt.show(block=True)