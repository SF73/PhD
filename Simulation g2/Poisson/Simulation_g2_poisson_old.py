# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:29:05 2019

@author: Sylvain.FINOT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
np.random.seed(12789749)
plt.style.use('Rapport')
plt.rc('axes', titlesize=12)
Rate = int(9.2e4+1.7e5) #nombre de coups/s avant beamsplitter
BS = 1.7e5/(9.2e4+1.7e5) #portion du signal qui va sur le detecteur (pas la clock)
T = 50 #nombre de secondes de simulation
dt = 512e-12 #binning en s
t = 0 #compteur de temps
clock = 0 #clock
delay = []
expected_count = (Rate*BS)*(Rate*(1-BS))*T

while(t<T):
    d = np.random.exponential(1/Rate) #temps entre 2 photons avant beamsplitter
    if np.random.rand() < BS:
        clock += d
        delay.append(clock)
    else:
        clock = 0
    t+=d

def normalisation(t,fclock,fcount,T,dt):
    return fcount*fclock*T*dt * np.exp(-fclock*t)
bins = np.arange(0,max(delay),dt)
hist, bins_edges = np.histogram(delay,bins=bins)

fig, ax = plt.subplots()
#axins = inset_axes(ax, width="50%", height="30%", loc=3)
t = np.linspace(0,max(delay),int(1e5))
ax.loglog(bins_edges[1:]*1e9,hist,label="Simulation histogrammer")
ax.loglog(bins_edges[1:]*1e9,expected_count*(1e-9) * np.diff(bins_edges*1e9)[0]*np.ones_like(bins_edges[1:]),label="Theory : Perfect g2")
ax.loglog(t*1e9,normalisation(t,(1-BS)*Rate,BS*Rate,T,dt),label="Theory : histogram")
#axins.loglog(bins_edges[1:]*1e9,hist,label="Simulation histogrammer")
#axins.loglog(bins_edges[1:]*1e9,expected_count*(1e-9) * np.diff(bins_edges*1e9)[0]*np.ones_like(bins_edges[1:]),label="Theory : Perfect g2")
#axins.loglog(t*1e9,normalisation(t,(1-BS)*Rate,BS*Rate,T,dt),label="Theory : histogram")

ax.legend()
plt.title("Rate before splitting %.2e s-1, BeamSplitter %.2f, T %d s"%(Rate,BS,T))
plt.xlabel("Delay (ns)")
plt.ylabel("Counts")
#plt.hist(delay,bins)