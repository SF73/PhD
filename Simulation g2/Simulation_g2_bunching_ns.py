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
I = 1e-12/1.602e-19 *1e-9 #nombre d'electrons incidents/ns (poisson)
G = int(400)#int(5e3/(3*3.47)) #nombre de photons generés par e-
tau = 3 #temps de vie en ns
BS = 0.5#9.2/(17+9.2)#0.5 #portion du signal qui va sur le detecteur
T = 1*1e9 #nombre de ns de mesure

# Photons collectés/photons générés avant beam splitter
# Produit de IQErad Extraction x Collection x Efficacité
efficiency = 5e-3

dt = 256e-3 #binning en ns

#Largeur de l'histogram
bins = np.arange(0,20000,dt)

#Delai supplémentaire entre detecteur et picoharp (cable)
delay = 100

# Nombre d'électrons générés par packet
# Donne la possibilité a un photon du paquet n d'arriver 
# après un photon du paquet n+PacketSize 
PacketSize = 1000

#ne pas changer
Ne = int(T*I) #nombre d'e simuler
# =============================================================================


print("======================")
print("Simulation g2 monoexponentiel\n")
print("Current :\t\t{:.1f}pA \t{:.2e}e-/s".format((I*1.602e-19/1e-12*1e9),I*1e9))
print("Photons / e- :\t\t{:d}".format(G))
print("Efficiency :\t\t{:.2f}".format(efficiency))
print("Lifetime :\t\t{:.2f}ns".format(tau))
print("Binning :\t\t{:.0f}ps".format(dt*1e3))
print("Duration :\t\t{:.0f}s".format(T*1e-9))
print("BeamSplitter :\t\t{:.2f}".format(BS))
print("<Nd> :\t\t{:.2e}cp/s".format(I*G*efficiency*BS*1e9))
print("<Nc> :\t\t{:.2e}cp/s".format(I*G*efficiency*(1-BS)*1e9))
print("Photons a simuler :\t\t{:.2e}".format(I*G*efficiency*T))

@njit
def generateBunches(I,PacketSize,tau,G,efficiency):
    te = np.random.exponential(1/I,PacketSize).cumsum()
    photonBunch = ((te.T+np.random.exponential(tau,(PacketSize,G)).T).T).flatten()
    photonBunch = photonBunch[np.random.rand(photonBunch.size)<efficiency]
    photonBunch.sort()
    return photonBunch

@njit
def generateBunches2(I,PacketSize,tau,G,efficiency):
    #Genere les temps d'arrivéé des electrons
    te = np.random.exponential(1/I,PacketSize).cumsum()
    test = np.random.binomial(G,efficiency,PacketSize)
    te = np.repeat(te,test)
    #tester vrai loop for voir meme prange
    photonBunch = te+np.random.exponential(tau,te.size)
    photonBunch.sort()
    return photonBunch

@njit(parallel=True)
def simulatenoCorrelation(I,PacketSize,tau,G,efficiency,delay):
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        photonBunch = generateBunches2(I,PacketSize,tau,G,efficiency)
        #Genere les temps ABSOLUS d'arrivee      
        r = np.random.rand(photonBunch.size)
        a = r<BS #photons qui vont au detecteur       
        if (a.sum()!=0): 
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


@njit(parallel=True)
def simulateCorrelation(I,Packetsize,tau,G,efficiency,delay):
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        photonBunch = generateBunches2(I,PacketSize,tau,G,efficiency)
        
        r = np.random.rand(photonBunch.size)
        clock = photonBunch[r>BS]
        detector = photonBunch[r<BS] + delay
        delays = np.zeros((len(clock),len(detector)))
        for i in prange(len(clock)):
            for j in prange(len(detector)):
                delays[i][j] = -clock[i]+detector[j]
        #delays = (-np.expand_dims(clock,-1)+np.expand_dims(detector,0)).flatten()
        hist += np.histogram(delays,bins)[0]
#        if (n % (Nloop//100)==0):
#            print(100*n/Nloop)
    return hist

start = time.time()
nocorrelation = simulatenoCorrelation(I,PacketSize,tau,G,efficiency,delay)
execution_time = (time.time()-start)*1e3
print(execution_time)
print((execution_time/int(Ne*G*efficiency))*1e3)
print((execution_time/int(Ne*G))*1e3)

correlation = simulateCorrelation(I,PacketSize,tau,G,efficiency,delay)
def normalisation(t,fclock,fcount,T,dt):
    return fcount*fclock*T*dt * np.exp(-fclock*np.abs(t))

def normalisation2(t,fclock,fcount,T,dt,tau):
    uncorrelated = fcount*fclock*T*dt * np.exp(-fclock*np.abs(t))
    correl = fcount*fclock*T*dt * np.exp(-t/tau)
    return uncorrelated+correl
norm1 = normalisation(bins[1:],(BS)*I*G*efficiency,(1-BS)*I*G*efficiency,T,dt)
norm2 = normalisation(bins[1:],(1-BS)*I*G*efficiency,(BS)*I*G*efficiency,T,dt)

expected = normalisation2(bins[1:],(1-BS)*I*G*efficiency,BS*I*G*efficiency,T,dt,tau)
fig, axes = plt.subplots(1,1,sharex=True)
axes.plot(bins[1:],nocorrelation/norm1)
axes.plot(bins[1:],correlation/norm1[0])
#axes.plot(bins[1:],np.ones_like(bins[1:])*norm1[0])
#axes.plot(bins[1:],norm1)
#axes.plot(bins[1:],norm2)
#fig.set_size_inches(6.1, 5.023)
#axes.set_xscale("log")
#axes.set_xlim(100,2e4)
#axes.set_ylim(0,500)