# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:13:39 2019

@author: Sylvain.FINOT
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
eV_To_nm = cst.c*cst.h/cst.e*1E9

class Spectrum():
    def __init__(self, wavelenghts,Imax=None):
        self.fig, self.ax = plt.subplots()
        self.wavelenghts = wavelenghts
        self.graph, = self.ax.plot(eV_To_nm/self.wavelenghts,np.zeros(self.wavelenghts.shape[0]))
        if Imax: self.ax.set_ylim((0,Imax))
        self.ax.set_xlabel('Energy (eV)')
        self.ax.set_ylabel('Intensity (a.u)')
        self.fig.subplots_adjust(top=0.885, bottom=0.125, left=0.13, right=0.94)
        self.ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        self.minbar = self.ax.axvline(eV_To_nm/self.wavelenghts.max())
        self.maxbar = self.ax.axvline(eV_To_nm/self.wavelenghts.min())    
        self.ax.format_coord = self.format_coord
        
    def format_coord(self,x, y):
            return f'x={x:1.4f}, y={y:1.0f}, lambda={eV_To_nm/x:1.2f}'
    def set_y(self,y):
        self.graph.set_ydata(y)
    def set_limitbar(self,minw,maxw):
        self.minbar.set_xdata(np.repeat(eV_To_nm/maxw,2))
        self.maxbar.set_xdata(np.repeat(eV_To_nm/minw,2))
    def update(self):
        self.fig.canvas.draw_idle()
    def plot_meanSpectrum(self,wavelenght,intensity):
        self.ax.plot(eV_To_nm/wavelenght,intensity,c="k",alpha=.5)