# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:29:05 2019

@author: Sylvain.FINOT
"""
from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import timeit
wdir = os.getcwd()
os.chdir(r"C:\Users\sylvain.finot\Cloud CNRS\Python")
# sys.path.insert(1, os.path.dirname(sys.path[0]))
from FileHelper import getListOfFiles
os.chdir(wdir)
# from ReadPhu import readphu
plt.style.use('Rapport')
plt.rc('axes', titlesize=12)



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
        self.tau = 0.198 #temps de vie en ns
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
        #self.stopAt = None
    def getbins(self):
        return np.arange(0,self.binNumber*self.dt*1e-3,self.dt*1e-3)
    def get_eRate(self):
        return (self.I/1.602e-19)*1e-21
    def tostring(self):
        header = ""
        for i, j in self.__dict__.items():
            header += "{}\t{}\n".format(i,j)
        return header
def simulate(param):
    electronRate = (param.I/1.602e-19)*1e-21 # pA to e/ns
    # =============================================================================
    print("======================")
    print("Simulation g2 monoexponentiel\n")
    print("Current :\t\t{:.1f}pA \t{:.2e}e-/ns".format(param.I,electronRate))
    print("Photons / e- :\t\t{:d}".format(param.G))
    print("Efficiency :\t\t{:.2e}".format(param.efficiency))
    print("Lifetime :\t\t{:.2f}ns".format(param.tau))
    print("Binning :\t\t{:.0f}ps".format(param.dt))
    print("Duration :\t\t{:.0e}s".format(param.T))
    print("BeamSplitter :\t\t{:.2f}".format(param.BS))
    print("Delay :\t\t\t{:.2f}".format(param.delay))
    print("DeadTime :\t\t{:.2f}".format(param.deadTime))
    #print("<Nd> :\t\t{:.2e}cp/s".format(I*G*efficiency*BS*1e9))
    #print("<Nc> :\t\t{:.2e}cp/s".format(I*G*efficiency*(1-BS)*1e9))
    #print("Photons a simuler :\t\t{:.2e}".format(I*G*efficiency*T))

    if param.deadTime == None:
        hist = simulatenoCorrelation(electronRate,param.PacketSize,param.getbins(),param.tau,param.G,param.efficiency,param.delay,param.BS,param.T,param.deadTime)
    else:
        hist = simulatenoCorrelationDeadTime(electronRate,param.PacketSize,param.getbins(),param.tau,param.G,param.efficiency,param.delay,param.BS,param.T,param.deadTime)
    return param.getbins()[:-1],hist
@njit
def generateBunches(electronRate,PacketSize,tau,G,efficiency):
    te = np.random.exponential(1/electronRate,PacketSize).cumsum()
    photonBunch = ((te.T+np.random.exponential(tau,(PacketSize,G)).T).T).flatten()
    photonBunch = photonBunch[np.random.rand(photonBunch.size)<efficiency]
    photonBunch.sort()
    return photonBunch

@njit
def generateBunches2(electronRate,PacketSize,tau,G,efficiency):
    #Genere les temps d'arrivéé des electrons
    # te = np.random.exponential(1/electronRate,PacketSize).cumsum()
    te = -1/electronRate*np.log1p(-np.random.random(PacketSize)).cumsum()
    #nombre de photons par electron en tenant compte de l efficacite
    test = np.random.binomial(G,efficiency,PacketSize)
    te = np.repeat(te,test)
    
    #tester vrai loop for voir meme prange
    photonBunch = te+np.random.exponential(tau,te.size)
    photonBunch.sort()
    return photonBunch

@njit
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

@njit
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


@njit
def correlate(clock,detector):
    delays = []
    j=len(detector)-1
    for i in range(len(clock)-1,-1,-1):
        while detector[j]-clock[i]>0:
            delays.append(detector[j]-clock[i])
            j-=1
            if j<0:break
        if j<0:break
    return np.array(delays)

@njit(parallel=True)
def simulatenoCorrelation(electronRate,PacketSize,bins,tau,G,efficiency,delay,BS,T):
    Ne = int(T*electronRate*1e9) #nombre d'e simuler
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        #525 µs
        photonBunch = generateBunches2(electronRate,PacketSize,tau,G,efficiency)
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
def simulatenoCorrelationDeadTime(electronRate,PacketSize,bins,tau,G,efficiency,delay,BS,T,deadTime):
    Ne = int(T*electronRate*1e9) #nombre d'e simuler
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        photonBunch = generateBunches2(electronRate,PacketSize,tau,G,efficiency)
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

            if np.max(hist)>65536:
                return hist
    
    return hist

@njit(parallel=True)
def simulateCorrelation(electronRate,PacketSize,bins,tau,G,efficiency,delay,BS,T):
    Ne = int(T*electronRate)
    hist = np.zeros(len(bins)-1)
    Nloop = int(Ne//PacketSize)
    for n in prange(Nloop):
        photonBunch = generateBunches2(electronRate,PacketSize,tau,G,efficiency)
        
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

# def currentRange():
#     path = r"C:\Users\sylvain.finot\Cloud Neel\Data\2020\2020-01-16 - T2594 - 300K\Wire 1\g2\Spot\TEST2"
#     files = getListOfFiles(path)
#     files = [f for f in files if f.endswith(".phu")]
#     current = {"1":0.0024,"2":0.0096,"3":0.0372,"4":0.121,"5":0.364,"5.5":0.405}
#     fig,ax=plt.subplots()
#     for i in [1,5]:
#         p=files[i]
#         spot = p.split('spot')[-1].split('_')[0].replace("-",".")
#         t,counts,tags=readphu(p)
#         tags = dict(tags)
#         SyncDivider = tags['HistResDscr_HWSyncDivider(0)']
#         SyncRate = tags['HistResDscr_SyncRate(0)']
#         CountRate = tags['HistResDscr_InputRate(0)']
#         BinningFactor = tags['HistResDscr_MDescBinningFactor(0)']
#         Resolution = tags['HistResDscr_MDescResolution(0)']
#         IntegrationTime = tags['HistResDscr_MDescStopAfter(0)']
#         BS = CountRate/(CountRate+SyncRate)
#         electronRate = current[spot]*1e-9*1e-9/1.602e-19
#         nocorrelation = simulatenoCorrelationDeadTime(electronRate,PacketSize,tau,G,efficiency,delay,BS,IntegrationTime,0)
#         print("Spot {} done {}/{}".format(spot, (i+1),len(files)))
#         ax.plot(bins[:-1],nocorrelation,".")

def normalisation(t,fclock,fcount,T,dt,delay=0):
    return fcount*fclock*T*dt * np.exp(-fclock*np.abs(t-delay))

def normalisation2(t,fclock,fcount,T,dt,tau):
    uncorrelated = fcount*fclock*T*dt * np.exp(-fclock*np.abs(t))
    correl = fcount*fclock*T*dt * np.exp(-t/tau)
    return uncorrelated+correl

def norm2(t,I,G,eff,BS,T,dt,tau,delay):
    eRate = (I/1.602e-19)*1e-21
    N = eRate*G*eff
    return N**2*(T*1e9)*(dt*1e-3)*BS*(1-BS)*np.exp(-N*(1-BS)*np.abs(t-delay))

def efficiency_vs_I():
    I = 1
    efficiency = 1e-2
    span = np.concatenate((np.round(np.linspace(2,400,6)),np.round(np.linspace(400,1500,3))))
    param = SimulationParameter()
    param.I = 1
    param.T = 10
    param.BS = 0.5
    param.G = 40
    param.efficiency = 1e-2
    for i in (I*span):
        param.I = i
        for j in (efficiency/span):
            param.efficiency = j
            bins,hist = simulate(param)
            name = "%s.dat"%("I%d_eff%.7f"%(i,j)).replace('.','-')
            np.savetxt(name,hist,header=param.tostring(),fmt="%d")
            # plt.plot(bins,hist,".",label="G{}_eff{}.dat".format(str(i).replace(".","-"),str(j).replace(".","-")))
def efficiency_vs_G():
    G = 10
    efficiency = 1e-2
    span = np.array([1,2,10,40,100])
    param = SimulationParameter()
    param.I = 10
    param.T = 10
    param.BS = 0.5
    for i in (G*span):
        param.G = i
        for j in (efficiency/span):
            param.efficiency = j
            bins,hist = simulate(param)
            #np.savetxt("I{}_eff{}.dat".format(str(i).replace(".","-"),str(j).replace(".","-")),hist)
            plt.plot(bins,hist,".",label="G{}_eff{}.dat".format(str(i).replace(".","-"),str(j).replace(".","-")))
            np.savetxt("GvsEff\G{}_eff{}.dat".format(str(i).replace(".","-"),str(j).replace(".","-")),hist,header=param.tostring(),fmt="%d")
            #plt.plot(bins,norm2(bins,param.I,param.G,param.efficiency,param.BS,param.T,4,param.tau,26))
def BeamSplitter():
    span = np.linspace(0.1,0.9,9)
    param = SimulationParameter()
    param.I = 10
    param.T = 10
    param.BS = 0.5
    param.G = 40
    param.efficiency = 1e-2
    for i in span:
        param.BS = i
        bins,hist = simulate(param)
        np.savetxt("BS{}.dat".format(str(i).replace(".","-")),hist)
        plt.plot(bins,hist,".",label="BS{}".format(str(i).replace(".","-")))
        
def lifetimeEffect():
    span = np.array([0.1,0.2,0.4,1,10])
    param = SimulationParameter()
    param.I = 10
    param.T = 100
    param.BS = 0.5
    param.G = 40
    param.efficiency = 1e-4
    for i in span:
        param.tau = i
        bins,hist = simulate(param)
        np.savetxt("tau{}.dat".format(str(i).replace(".","-")),hist)
        plt.plot(bins,hist,".",label="BS{}".format(str(i).replace(".","-")))
        
def deadTimeEffect():
    span = np.array([200]) #en ns
    param = SimulationParameter()
    param.I = 10
    param.T = 100
    param.BS = 0.5
    param.G = 40
    param.efficiency = 1e-3
    param.tau = 0.2
    
    for i in span:
        param.deadTime = i
        bins,hist = simulate(param)
        np.savetxt("deadTime{}.dat".format(str(i).replace(".","-")),hist)
        #plt.plot(bins,hist,".",label="BS{}".format(str(i).replace(".","-")))
def CurrentEffect():
    span = np.concatenate((np.linspace(1,20,20),np.linspace(25, 100,26)))
    param = SimulationParameter()
    param.I = 1
    param.T = 10
    param.BS = 0.5
    param.G = 40
    param.efficiency = 1e-2
    param.tau = 0.2
    
    for i in span:
        param.I = i
        bins,hist = simulate(param)
        np.savetxt("I\I{}.dat".format(str(i).replace(".","-")),hist,header=param.tostring(),fmt="%d")
        #plt.plot(bins,hist,".",label="BS{}".format(str(i).replace(".","-")))

def timeeffect():
    param = SimulationParameter()
    ratio = []
    for T in np.logspace(-1,1,20):
        param.I = 10
        param.T = T
        param.G = 40
        param.tau = 0.2
        param.BS = 0.5
        param.efficiency = 1e-2
        bins,hist = simulate(param)
        ratio.append(np.max(hist)/np.mean(hist[:5000]))
    print(np.mean(ratio))
    print(np.std(ratio))
def Geffdead_vs_I():
    directory = r"C:\Users\sylvain.finot\Cloud CNRS\Python\Simulation g2\Geffdead_vs_I"
    param = SimulationParameter()
    for eff in np.linspace(0.0425/40,0.38/40,11):
        for i in np.linspace(2,12,16):
            param.I = i
            param.T = 50
            param.G = 40
            param.tau = 0.2
            param.BS = 0.5
            param.efficiency = eff
            bins, hist = simulate(param)
            np.savetxt("Geffdead_vs_I\Geff%.7f_I%.4f.dat"%(40*eff,i),hist,header=param.tostring(),fmt="%d")
def Geff():
    directory = r"C:\Users\sylvain.finot\Cloud CNRS\Python\Simulation g2\Geff"
    param = SimulationParameter()
    for eff in np.linspace(0.001,0.01,11):
        for G in [4,40,400]:
            param.I = 2
            param.T = 50
            param.G = G
            param.tau = 0.2
            param.BS = 0.5
            param.efficiency = eff
            bins, hist = simulate(param)
            np.savetxt("Geff\G%.7f_eff%.4f.dat"%(G,eff),hist,header=param.tostring(),fmt="%d")

        
def generateTrainingData():
    directory = r"C:\Users\sylvain.finot\Cloud CNRS\Python\Simulation g2\trainingData"
    files = getListOfFiles(directory)
    param = SimulationParameter()
    Irange = [1,150]
    Trange = [1,10]
    BSrange = [10,90]
    Grange = [2,480]
    effrange = [1e-4,1e-2]
    taurange = [90,600] #ps
    N = 100
    for i in range(len(files),N+len(files)):
        param.T=np.random.random_sample()*(max(Trange)-min(Trange))+min(Trange)
        param.I = np.random.randint(*Irange)
        param.G = np.random.randint(*Grange)
        param.tau = np.random.randint(*taurange)/1000
        param.BS = np.random.randint(*BSrange)/100
        param.efficiency = np.random.random_sample()*(max(effrange)-min(effrange))+min(effrange)
        bins, hist = simulate(param)
        np.savetxt("trainingData\%u.dat"%i,hist,header=param.tostring(),fmt="%d")
        