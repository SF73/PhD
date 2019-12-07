# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:29:05 2019

@author: Sylvain.FINOT
"""
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use('Rapport')
plt.rc('axes', titlesize=12)

# =============================================================================
#Simulation g2 monoexponentiel
#Parametres
I = 4e-12/1.602e-19 #nombre d'electrons incidents/s (poisson)
G = int(5)#5e3/(3*3.47) #nombre de photons generés par e-
tau = 0.2e-9 #temps de vie en s
BS = 0.5 #portion du signal qui va sur le detecteur
T = 50 #nombre de secondes de mesure

# Photons collectés/photons générés avant beam splitter
# Produit de Extraction x Collection x Efficacité
efficiency = 1e-2

dt = 4e-12 #binning en s

#Largeur de l'histogram
bins = np.arange(0,200e-9,dt)

#Delai supplémentaire entre detecteur et picoharp (cable)
delay = 100e-9

# Nombre d'électrons générés par packet
# Donne la possibilité a un photon du paquet n d'arriver 
# après un photon du paquet n+PacketSize 
PacketSize = 10000

#ne pas changer
Ne = int(T*I) #nombre d'e simuler
# =============================================================================


print("======================")
print("Simulation g2 monoexponentiel\n")
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
        a = r<BS #photons qui vont au detecteur       
        if (a.sum()==0):
            continue #pas de photons conservés
        photonBunch[a] += delay
        idx = np.argsort(photonBunch)
        photonBunch = photonBunch[idx]
        a = a[idx]
        k = ~a
        #delai entre chaque photon
        relative_delay = np.convolve(photonBunch,[1,-1])[:-1]
        
        #remise a 0 a chaque start
        c = np.cumsum(a*relative_delay)
        d = np.diff(np.concatenate((np.array([0.]), c[k])))
        relative_delay[k] = -d
        
        hist += np.histogram(np.cumsum(relative_delay)[a],bins)[0]
    
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

start = time.time()
nocorrelation = simulatenoCorrelation(I,PacketSize,tau,G,efficiency,delay)
execution_time = (time.time()-start)*1e3
print(execution_time)
print((execution_time/int(Ne))*1e3)

def normalisation(t,fclock,fcount,T,dt):
    return fcount*fclock*T*dt * np.exp(-fclock*np.abs(t))
norm1 = normalisation(bins[1:]-delay,(1-BS)*I*G*efficiency,BS*I*G*efficiency,T,dt)
norm2 = normalisation(bins[1:]-delay,(BS)*I*G*efficiency,(1-BS)*I*G*efficiency,T,dt)
expected = normalisation(0,(1-BS)*I*G*efficiency,BS*I*G*efficiency,T,dt)
fig, axes = plt.subplots(1,1,sharex=True)
axes.plot(bins[1:]*1e9,nocorrelation)
axes.plot(bins[1:]*1e9,norm2)


#fig.set_size_inches(6.1, 5.023)
#axes.set_xscale("log")
#axes.set_xlim(100,2e4)
#axes.set_ylim(0,500)