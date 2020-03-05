# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:29:05 2019

@author: Sylvain.FINOT
"""
from numba import njit, prange, jit
import numpy as np
import time
#import matplotlib.pyplot as plt
#import os
#wdir = os.getcwd()
#os.chdir(r"C:\Users\sylvain.finot\Cloud CNRS\Python")
# sys.path.insert(1, os.path.dirname(sys.path[0]))
#from FileHelper import getListOfFiles
#os.chdir(wdir)
# from ReadPhu import readphu
# plt.style.use('Rapport')
# plt.rc('axes', titlesize=12)



class SimulationParameter:
    '''Class to hold all the parameters
    
    Parameters
    ----------
    I : float
        Ebeam current in pA
    G : int
        number of electron-hole pair per electron
    tau : float
        lifetime in ns
    T : float
        Integration time in s
    deadTime : float
        dead-time of the TCSPC in ns
    efficiency : float
        photon collected/photon generated
    BS : float
        Ndetector/(Ndetector+Nclock)
    dt : bin\'s width in ps
    delay : float
        delay between the two detectors in ns
    PacketSize : int
        number of electron to generate per loop 
    '''
    def __init__(self):
        self.I = 2# in pA
        self.G = int(400)#int(5e3/(3*3.47)) #nombre de photons generés par e-
        self.tau = 0.200 #temps de vie en ns
        self.BS = 0.5 #portion du signal qui va sur le detecteur
        self.T = 1 #nombre de s de mesure
        self.deadTime = 86
        # Photons collectés/photons générés avant beam splitter
        # Produit de IQErad x Extraction x Collection x Efficacité
        self.efficiency = 1e-3      
        self.dt = 4 #binning en ps        
        #Largeur de l'histogram
        self.binNumber = 2**16
        #Delai supplémentaire entre detecteur et picoharp (cable)
        self.delay = 26#100        
        # Nombre d'électrons générés par packet
        # Donne la possibilité a un photon du paquet n d'arriver 
        # après un photon du paquet n+PacketSize 
        self.PacketSize = 10000
        self.stopAt = 10000
    def getbins(self):
        return np.arange(0,self.binNumber*self.dt*1e-3,self.dt*1e-3)
    def get_eRate(self):
        return (self.I/1.602e-19)*1e-21
    def tostring(self):
        header = ""
        for i, j in self.__dict__.items():
            header += "{}\t{}\n".format(i,j)
        return header
def simulate(param,fast=False):
    electronRate = (param.I/1.602e-19)*1e-21 # pA to e/ns
    # =============================================================================
    # print("======================")
    # print("Simulation g2 monoexponentiel\n")
    # print("Current :\t\t{:.1f}pA \t{:.2e}e-/ns".format(param.I,electronRate))
    # print("Photons / e- :\t\t{:d}".format(param.G))
    # print("Efficiency :\t\t{:.2e}".format(param.efficiency))
    # print("Lifetime :\t\t{:.2f}ns".format(param.tau))
    # print("Binning :\t\t{:.0f}ps".format(param.dt))
    # print("Duration :\t\t{:.0e}s".format(param.T))
    # print("BeamSplitter :\t\t{:.2f}".format(param.BS))
    # print("Delay :\t\t\t{:.2f}".format(param.delay))
    # print("DeadTime :\t\t{:.2f}".format(param.deadTime))
    #print("<Nd> :\t\t{:.2e}cp/s".format(I*G*efficiency*BS*1e9))
    #print("<Nc> :\t\t{:.2e}cp/s".format(I*G*efficiency*(1-BS)*1e9))
    #print("Photons a simuler :\t\t{:.2e}".format(I*G*efficiency*T))

    if fast:
        lookupTable = genLookupTable(param.G,param.efficiency)
        p = 1-(1-param.efficiency)**param.G
        param.PacketSize = np.round(1600/p)
        hist = simulatenoCorrelationDeadTime2(electronRate,param.PacketSize,param.getbins(),param.tau,param.G,param.efficiency,param.delay,param.BS,param.T,param.deadTime,param.stopAt,lookupTable)
    else:
        hist = simulatenoCorrelationDeadTime(electronRate,param.PacketSize,param.getbins(),param.tau,param.G,param.efficiency,param.delay,param.BS,param.T,param.deadTime,param.stopAt)
    return param.getbins()[:-1],hist

@njit(cache=True)
def generateBunches(electronRate,PacketSize,tau,G,efficiency):
    """
    

    Parameters
    ----------
    electronRate : TYPE
        DESCRIPTION.
    PacketSize : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    G : TYPE
        DESCRIPTION.
    efficiency : TYPE
        DESCRIPTION.

    Returns
    -------
    photonBunch : TYPE
        DESCRIPTION.

    """
    #Genere les temps d'arrivéé des electrons
    # te = np.random.exponential(1/electronRate,PacketSize).cumsum()
    te = (-1/electronRate*np.log1p(-np.random.random(PacketSize))).cumsum()
    #nombre de photons par electron en tenant compte de l efficacite
    test = np.random.binomial(G,efficiency,PacketSize)
    te = np.repeat(te,test)
    
    #tester vrai loop for voir meme prange
    # photon = -1/tau*np.log1p(-np.random.random(te.size))
    # photon.sort()
    # photonBunch = te+photon
    photonBunch = te+np.random.exponential(tau,te.size)
    photonBunch.sort()
    return photonBunch

@njit
def generateBunches3(electronRate,PacketSize,tau,G,efficiency):
    test = np.random.binomial(G,efficiency,PacketSize)
    idx = np.where(test>0)[0]
    a = np.ones(idx.size)
    a[1:] = np.diff(idx)
    te = (-a/(electronRate)*np.log1p(-np.random.random(a.size))).cumsum()
    te = np.repeat(te,test[idx])
    photonBunch = te+np.random.exponential(tau,te.size)
    photonBunch.sort()
    return photonBunch


def Cnk(n,k):
    return np.math.factorial(n)//(np.math.factorial(k)*np.math.factorial(n-k))

def posBin(k,n,p):
    return Cnk(n,k)*np.power(p,k)*np.power((1-p),(n-k))/(1-np.power((1-p),n))

def Bin(k,n,p):
    # return Cnk(n,k)*np.power(p,k)*np.power((1-p),(n-k))
    return np.power(p,k)*np.power((1-p),(n-k))
def genLookupTable(n,p):
    lookupTable = [posBin(k,n,p)for k in np.arange(1,n+1,dtype=np.float64)]
    # lookupTable = np.array([0]+lookupTable)
    lookupTable = np.cumsum([0.]+lookupTable)
    lookupTable /= lookupTable[-1]
    idx = np.where(lookupTable<1)[0].max()
    lookupTable=lookupTable[0:idx+2]
    return lookupTable

@njit(cache=True)
def TruncBin(lookupTable,size):
    return np.searchsorted(lookupTable,np.random.rand(int(size)))

@njit(cache=True)
def generateBunches2(electronRate,PacketSize,tau,G,efficiency,lookupTable):
    p = 1-(1-efficiency)**G
    eRate = electronRate * p
    
    te = (-1/eRate*np.log1p(-np.random.random(np.random.binomial(PacketSize,p)))).cumsum()
    #nombre de photons par electron en tenant compte de l efficacite
    # Garray = TruncBin(lookupTable,te.size)
    photonBunch = np.repeat(te,TruncBin(lookupTable,te.size))
    

    photonBunch += (-tau*np.log1p(-np.random.random(photonBunch.size)))#np.random.exponential(tau,te.size)
    photonBunch.sort()
    # photonBunch = te+np.random.exponential(tau,te.size)
    # photonBunch.sort()
    return photonBunch
@njit(cache=True)
def applyDeadTime2(array,deadTime):
    i=0
    if len(array)<2:
        return array
    onTime=[array[0]]
    while i < (len(array)-1):
        for j in range(i+1,len(array)):
            #pour chaque element on regarde quand le dead time est fini
            if (array[j]>(array[i]+deadTime)):
                #delai plus grand que deadTime
                #on garde le delai
                onTime.append(array[j])
                #on saute les photons compris dans le deadTime
                i=j
                break
        if (j==len(array)-1):
            break
    return np.array(onTime)

@njit(cache=True)
def applyDeadTime(array,deadTime):
    i=0
    c=0
    if len(array)<2:
        return array
    onTime=np.empty_like(array)
    onTime[0]=array[0]
    while i < (len(array)-1):
        for j in range(i+1,len(array)):
            #pour chaque element on regarde quand le dead time est fini
            if (array[j]>(array[i]+deadTime)):
                #delai plus grand que deadTime
                #on garde le delai
                c = c+1
                onTime[c]=array[j]
                #on saute les photons compris dans le deadTime
                i=j
                break
        if (j==len(array)-1):
            break
    return onTime[0:c+1]


@njit(cache=True)
def correlate(clock,detector):
    delays = []
    j=len(detector)-1
    for i in range(len(clock)-1,-1,-1):
        #On parcourt les clocks antichronologiquement pour trouver la dernière
        while detector[j]-clock[i]>0:
            #On regarde les photons du detecteur antichronologiquement
            #On ajoute les différences de temps jusqu'à trouver un photon detecteur
            #anterieur à la clock actuelle
            delays.append(detector[j]-clock[i])
            j-=1
            if j<0:break
        #detecteur anterieur à la clock => on passe a la clock précédente
        if j<0:break
    return np.array(delays)

@njit(parallel=True)
def simulatenoCorrelation(electronRate,PacketSize,bins,tau,G,efficiency,delay,BS,T):
    Ne = int(T*electronRate*1e9) #nombre d'e simuler
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        #525 µs
        photonBunch = generateBunches(electronRate,PacketSize,tau,G,efficiency)
        #Genere les temps ABSOLUS d'arrivee      
        # r = np.random.rand(photonBunch.size)
        # a = r<BS #photons qui vont au detecteur
        #34.7 µs
        a=np.random.rand(photonBunch.size)<BS
        #a=np.random.binomial(1,BS,photonBunch.size)
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

@njit(parallel=True,cache=True)
def simulatenoCorrelationDeadTime(electronRate,PacketSize,bins,tau,G,efficiency,delay,BS,T,deadTime,stopAt):
    Ne = int(T*electronRate*1e9) #nombre d'e simuler
    hist = np.zeros(len(bins)-1,dtype=np.int16)
    Nloop = int(np.round(Ne/PacketSize))
    for n in prange(Nloop):
        photonBunch = generateBunches(electronRate,PacketSize,tau,G,efficiency)
        a=np.random.rand(photonBunch.size)<BS
        #photons qui vont au detecteur
        #a=np.random.binomial(1,BS,photonBunch.size)
        if (a.sum()!=0):
            photonBunch[a] += delay
            clock = photonBunch[~a]
            detector = photonBunch[a]
            clock = applyDeadTime(clock,deadTime)
            detector = applyDeadTime(detector,deadTime)
            delays = correlate(clock,detector)
            hist += np.histogram(delays,bins)[0]

            if np.max(hist)>stopAt:
                return hist
    
    return hist

@njit(parallel=True,cache=True)
def simulatenoCorrelationDeadTime2(electronRate,PacketSize,bins,tau,G,efficiency,delay,BS,T,deadTime,stopAt,lookupTable):
    Ne = int(T*electronRate*1e9) #nombre d'e simuler
    hist = np.zeros(len(bins)-1,dtype=np.int16)
    Nloop = int(np.round(Ne/PacketSize))
    for n in prange(Nloop):
        photonBunch = generateBunches2(electronRate,PacketSize,tau,G,efficiency,lookupTable)
        #120ms
        a=np.random.rand(photonBunch.size)<BS
        #120ms
        # #photons qui vont au detecteur
        # #a=np.random.binomial(1,BS,photonBunch.size)
        if (a.sum()!=0):
            photonBunch[a] += delay
            clock = photonBunch[~a]
            detector = photonBunch[a]
            #120
            clock = applyDeadTime(clock,deadTime)
            detector = applyDeadTime(detector,deadTime)
            #154
            delays = correlate(clock,detector)
            #170
            hist += np.histogram(delays,bins)[0]
            #240ms

            if np.max(hist)>stopAt:
                return hist
    
    return hist


@njit(parallel=True)
def simulateCorrelation(electronRate,PacketSize,bins,tau,G,efficiency,delay,BS,T):
    Ne = int(T*electronRate)
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        photonBunch = generateBunches(electronRate,PacketSize,tau,G,efficiency)
        
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

def test(fast=True):
    param = SimulationParameter()
    param.I = 300
    param.T = 10
    param.BS = 0.5
    param.G = 10
    param.dt = 4
    param.efficiency = 1e-3
    param.deadTime = 86#0.00001
    start = time.time()
    bins,hist = simulate(param,fast)
    stop = time.time()
    print(stop-start)
    import matplotlib.pyplot as plt
    plt.plot(bins,hist)