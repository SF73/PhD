# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:29:05 2019

@author: Sylvain.FINOT
"""
from numba import jit, njit, prange
import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use('Rapport')
plt.rc('axes', titlesize=12)
I = (17+9.2)*1e4#4e-12/1.602e-19 #nombre d'electron/s incidents (poisson)
G = int(1)#5e3/(3*3.47) #a estimer avec CASINO
tau = 0*0.2e-9 #temps de vie de l'emetteur en s
BS = 9.2/(17+9.2) #portion du signal qui va sur le detecteur (pas la clock)
T = 50 #nombre de secondes de simulation
efficiency = 1 #photons collectés/photons générés avant beam splitter
dt = 512e-12 #binning en s
#bins = np.arange(0,2**16)*dt
bins = np.arange(0,20000e-9,dt)
delay = 0e-9
#expected_count = (Rate*BS)*(Rate*(1-BS))*T
Ne = int(T*I) #nombre d'e simuler
PacketSize = 1000
print("======================")
print("Simulation\n")
print("Current :\t\t{:.1f}pA \t{:.2e}e-/s".format((I*1.602e-19/1e-12),I))
print("Photons / e- :\t\t{:d}".format(G))
print("Efficiency :\t\t{:.2f}".format(efficiency))
print("Lifetime :\t\t{:.2f}ns".format(tau*1e9))
print("Binning :\t\t{:.0f}ps".format(dt*1e12))
print("Duration :\t\t{:.0f}s".format(T))
print("BeamSplitter :\t\t{:.2f}".format(BS))
print("<Nd> :\t\t{:.2e}cp/s".format(I*G*efficiency*BS))
print("<Nc> :\t\t{:.2e}cp/s".format(I*G*efficiency*(1-BS)))

@njit
def generateBunches(I,PacketSize,tau,G,efficiency):
    te = np.random.exponential(1/I,PacketSize).cumsum()
    photonBunch = ((te.T+np.random.exponential(tau,(PacketSize,G)).T).T).flatten()
    photonBunch = photonBunch[np.random.rand(photonBunch.size)<efficiency]
    photonBunch.sort()
    return photonBunch


@njit(parallel=True)
def simulatenoCorrelation(I,PacketSize,tau,G,efficiency,delay):
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        photonBunch = generateBunches(I,PacketSize,tau,G,efficiency)
        #Genere les temps ABSOLUS d'arrivee
        
        r = np.random.rand(photonBunch.size)
        k = r>BS #photons qui vont a l'horloge
        detector = ~k #photons qui vont au detecteur
        
        if (detector.sum()==0):
            continue #pas de photons conservés

        photonBunch[detector] += delay # ajout du delai au detecteur
        
        #sort
        idx = np.argsort(photonBunch)
        photonBunch = photonBunch[idx]
        detector = detector[idx]
        
        #supprimer tous ceux avant le 1er start
        idfirststart = np.where(~detector)[0][0]
        detector = detector[idfirststart:]
        photonBunch = photonBunch[idfirststart:]
        
        #supprimer tous ceux avant le dernier start-stop
#        idlaststop = np.where(detector)[0][-1]
#        detector = detector[:idlaststop]
#        photonBunch = photonBunch[:idlaststop]
        
        start = np.diff(np.concatenate((np.array([1]),detector*1)))
#        
        #Detection start stop
        idstartstop = np.where(start==1)[0]
        #delays = photonBunch[idstartstop]-photonBunch[idstartstop+1]

        #Detection stop start
        idstopstart = np.where(start==-1)[0]
        #delays = photonBunch[idstartstop+1]-photonBunch[idstartstop]
        
        if (idstopstart[-1]>idstartstop[-1]):idstopstart=idstopstart[:-1]
        delays = np.abs(photonBunch[idstartstop]-photonBunch[idstopstart])
        hist += np.histogram(delays,bins)[0]
    
    return hist


@njit(parallel=False)
def simulateCorrelation(I,Packetsize,tau,G,efficiency,delay):
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        photonBunch = generateBunches(I,PacketSize,tau,G,efficiency)
        
        r = np.random.rand(photonBunch.size)
        clock = photonBunch[r>BS]
        detector = photonBunch[r<BS] + delay
        delays = (-np.expand_dims(clock,-1)+np.expand_dims(detector,0)).flatten()
        hist += np.histogram(delays,bins)[0]
        if (n % (Nloop//100)==0):
            print(100*n/Nloop)
    return hist

#@njit(parallel=False)
#def test():
#    test = int(1e6)
#    for i in prange(test):
#        foo()
#@njit(parallel=False)
#def foo():
#    clock = np.array([1,4,5,7,11,15,19])
#    detector = np.array([2,3,6,10,13,20])
#    test = np.expand_dims(clock,1)
#    test2 = np.expand_dims(detector,0)
#    return test + test2

start = time.time()
nocorrelation = simulatenoCorrelation(I,PacketSize,tau,G,efficiency,delay)
execution_time = (time.time()-start)*1e3
print(execution_time)
print((execution_time/int(Ne//PacketSize))*1e3)
#start = time.time()
#correlate =  simulateCorrelation(I,PacketSize,tau,G,efficiency,delay)
#execution_time = (time.time()-start)*1e3
#print(execution_time)
#print((execution_time/int(Ne//PacketSize))*1e3)

def normalisation(t,fclock,fcount,T,dt):
    return fcount*fclock*T*dt * np.exp(-fclock*np.abs(t))
norm = normalisation(bins[1:]-delay,(1-BS)*I*G*efficiency,BS*I*G*efficiency,T,dt)
expected = normalisation(0,(1-BS)*I*G*efficiency,BS*I*G*efficiency,T,dt)
#plt.plot(bins[1:]*1e9,norm,label="Theory : histogram")
#axes[0].plot(bins[1:]*1e9,correlate)
fig, axes = plt.subplots(1,1,sharex=True)
fig.set_size_inches(6.1, 5.02)
#axes[0].plot(bins[1:]*1e9,nocorrelation/norm)
#axes[0].plot(bins[1:]*1e9,nocorrelation/np.mean(nocorrelation[-50:]))
#axes[1].plot(bins[1:]*1e9,nocorrelation)
axes.plot(bins[1:]*1e9,norm)
axes.plot(bins[1:]*1e9,nocorrelation)
axes.set_xscale("log")
axes.set_xlim(100,2e4)
axes.set_ylim(0,500)
#axes.plot(bins[1:]*1e9,correlate,alpha=0.5)

