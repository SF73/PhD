# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:19:22 2019

@author: Sylvain.FINOT
"""

import numpy as np
from scipy.integrate import odeint
from scipy import signal
import matplotlib.pyplot as plt


# function that returns dn/dt
taueff = 80e-9 * 1e9

A=3.7e6* 1e-9#1/taueff
B=1e-11* 1e-9
C=2.6e-31* 1e-9
Ibeam = 5E3/(3*3.47) * 1E-9/1.602E-19 / (4/3*np.pi*(100e-9)**3) / 1e6 * 1e-9#G cm**3/ns
dt = 64e-12 * 1e9
# initial condition
y0 = 0

# time points
t = np.arange(0,2**16)*dt
sigma = 2e-9*1e9
gx = np.arange(-3*sigma, 3*sigma, dt)
gaussian = np.exp(-(gx/sigma)**2/2)
def beamblanker(f,duty,t,Min, Max):
    amplitude = (Max-Min)/2
    offset = (Max+Min)/2
    u = amplitude*(signal.square(2*np.pi*f*1e-9*(t-0.25/(f*1e-9)),duty))+offset 
    result = np.convolve(u, gaussian, mode="same")/np.sum(gaussian)
    return u#result

u = beamblanker(4e5,0.5,t,Min=0,Max=1)
def model(n,t):
    if int(t/dt)>=2**16:
        mod = 0
    else:
        mod = u[int(t/dt)]
    dydt = -n/taueff + mod*1e15#Ibeam
    return dydt

def N(t,A,B,n0):
    return n0*A/(n0*B*(np.exp(A*t)-1)+A*np.exp(A*t))

def ABC(n,t):
    if int(t/dt)>=2**16:
        mod = 0
    else:
        mod = u[int(t/dt)]
    return -A*n-B*n**2-C*n**3# + mod*Ibeam
    # solve ODE
fig, ax = plt.subplots()
for n0 in np.logspace(12,16,5):
    y = odeint(ABC,n0,t)
    color = next(ax._get_lines.prop_cycler)['color']
    # plot results
    ax.plot(t,(y/max(y)),color,label='n0 : %.1e'%n0)
    ax.plot(t,N(t,A,B,n0)/n0,color,'--',label='A/Bn0 : %.1e'%(3.7e6/(1e-11*n0)))
    ax.plot(np.linspace(1e-2,200,1000),1/(1+(A+n0*B)*np.linspace(1e-3,200,1000)),'r')
    ax.plot(np.linspace(1e-2,1000,5000),A/(A+n0*B)*np.exp(-A*np.linspace(1e-2,1000,5000)),'g')
    ax.set_ylabel('values')
    ax.set_xlabel('time')
    ax.legend()
    ax.set_xlim(-10,200)
    ax.set_ylim(1E-3,1.2)
    ax.set_yscale('log')
ax.plot(t,N(t,3.7e6 * 1e-9,0*1e-11 * 1e-9,n0)/n0,color,'-.')

    #plt.plot(t*1e9,u,label='Beam Blanker Opacity')


#y = odeint(model,0,t)
#plt.plot(t,(y/max(y)),color,label='simu')
#plt.plot(t,u,label='Beam Blanker Opacity')
#def read(path):
#    counts = np.loadtxt(path) #Full histogram
#    binNumber = int(counts[0]) #Nombre de bin
#    binsize = counts[3] #Taille d'un bin
#    counts = counts[-binNumber:] #histogram
##    counts = counts/max(counts)
#    t = np.arange(binNumber)*binsize - 33#echelle de temps en ns
#    return t, counts
#t,counts = read(r"X:\Sylvain\2019-10-07 - T2455 - 300K\TRCL1_440nm_slit0-5_HV5kV_spot4_f400kHz_pulse1-25us.dat")
#
#plt.plot(t+625,counts/(0.97*max(counts)),'k',label='data')