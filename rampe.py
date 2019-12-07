# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:29:37 2019

@author: Sylvain
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:19:50 2019

@author: sylvain.finot
"""

from expfit import process_fromFile
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools
#import pickle
plt.style.use('Rapport')
plt.rc('axes', labelsize=16)
#plt.rc('font', family='serif',size=14)
#plt.rc('text', usetex=False)
#plt.rc('xtick', labelsize=14)
#plt.rc('ytick', labelsize=14)
#plt.rc('axes', labelsize=14)

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


fig, ax = plt.subplots()
fig.patch.set_alpha(0)
markers = itertools.cycle(["o",'D',"s",'h','H','8','*'])
files=getListOfFiles(r"C:\Users\sylvain.finot\Cloud Neel\Data")
# =============================================================================
# T2594
mask = [x for x in files if ((("TRCL" in x) & (x.endswith(".dat")))& ("T2594"in x) & (not ("Rampe"in x)) & (not("Al" in x)) & (not("Ag" in x)))]
T = list()
Tau = list()
T10 = list()
for p in mask:
    idx = p.find('K')
    Temp = int(p[idx-3:idx])
    A,tau,t10,A1,A2,tau1,tau2,R = process_fromFile(p,save=False,autoclose=True,merge=False)
    if ((1-R)>1e-2):continue
    taueff = -(A1*tau1+A2*tau2)/(A1+A2)
    T.append(Temp)
    Tau.append(tau)
    T10.append(t10)
T = np.array(T)
Tau = np.array(Tau)*1e3
T10 = np.array(T10)*1e3
m = next(markers)
#bx.plot(10*np.ones(len(T10)),T10,m,alpha=0.6,label='T2594')
ax.plot(T,T10,m,alpha=0.6,label='T2594 - different wires')
#ax.errorbar(T,Tau,yerr=25,fmt=m,label='T2594 - different wire')
#ax.errorbar([5,300],[240,161],yerr=25,fmt=m,label='T2594')
# =============================================================================
# =============================================================================
# T2594
mask = [x for x in files if ((("TRCL" in x) & (x.endswith(".dat")))& ("T2594"in x) & ("Rampe"in x))]
T = list()
Tau = list()
T10 = list()
for p in mask:
    idx = p.find('K')
    Temp = int(p[idx-3:idx])
    A,tau,t10,A1,A2,tau1,tau2,R = process_fromFile(p,save=False,autoclose=True,merge=True)
    if ((1-R)>1e-3):continue
    taueff = -(A1*tau1+A2*tau2)/(A1+A2)
    T.append(Temp)
    Tau.append(tau)
    T10.append(t10)
T = np.array(T)
Tau = np.array(Tau)*1e3
T10 = np.array(T10)*1e3
m = next(markers)
#ax.plot(T,Tau,m,alpha=0.6,label='T2594 - same wire')
#ax.errorbar(T,Tau,yerr=25,fmt=m,label='T2594 - same wire')
#ax.plot(T,Tau,m,alpha=0.6,label='T2594 same wire')
ax.plot(T,T10,m,alpha=0.6,label='T2594 same wire')
# =============================================================================
# =============================================================================
# T2597
mask = [x for x in files if ((("TRCL" in x) & (x.endswith(".dat")))& ("T2597"in x))]
T = list()
Tau = list()
T10 = list()
for p in mask:
    idx = p.find('K')
    Temp = int(p[idx-3:idx])
    A,tau,t10,A1,A2,tau1,tau2,R = process_fromFile(p,save=False,autoclose=True,merge=False)
    if ((1-R)>1e-3):continue
    taueff = -(A1*tau1+A2*tau2)/(A1+A2)
    T.append(Temp)
    Tau.append(taueff)
    T10.append(t10)
T = np.array(T)
Tau = np.array(Tau)*1e3
T10 = np.array(T10)*1e3
m = next(markers)
#ax.errorbar(T,Tau,yerr=25,fmt=m,label='T2597')
ax.plot(T,T10,m,alpha=0.6,label='T2597')
#ax.errorbar([5,300],[246,126],yerr=25,fmt=m,label='T2597')
# =============================================================================
## =============================================================================
## T2601
#mask = [x for x in files if ((("TRCL" in x) & (x.endswith(".dat")))& ("T2601"in x) & (not ("375nm" in x)))]
#T = list()
#Tau = list()
#T10 = list()
#for p in mask:
#    idx = p.find('K')
#    Temp = int(p[idx-3:idx])
#    A,tau,t10,A1,A2,tau1,tau2,R = process_fromFile(p,save=False,autoclose=True,merge=False)
#    if ((1-R)>1e-3):continue
#    taueff = -(A1*tau1+A2*tau2)/(A1+A2)
#    T.append(Temp)
#    Tau.append(taueff)
#    T10.append(t10)
#T = np.array(T)
#Tau = np.array(Tau)*1e3
#T10 = np.array(T10)*1e3
#m = next(markers)
#ax.plot(30*np.ones(len(Tau)),Tau,m,alpha=0.6,label='T2601')
#bx.plot(30*np.ones(len(T10)),T10,m,alpha=0.6,label='T2601')
## =============================================================================


## T2594 Al
#mask = [x for x in files if ((("TRCL" in x) & (x.endswith(".dat")))& ("T2594"in x) & (not ("Rampe"in x)) & ("Al" in x))]
#T = list()
#Tau = list()
#T10 = list()
#for p in mask:
#    idx = p.find('K')
#    Temp = int(p[idx-3:idx])
#    A,tau,t10,A1,A2,tau1,tau2,R = process_fromFile(p,save=False,autoclose=True,merge=True)
#    if ((1-R)>1e-2):continue
#    taueff = -(A1*tau1+A2*tau2)/(A1+A2)
#    T.append(Temp)
#    Tau.append(taueff)
#    T10.append(t10)
#T = np.array(T)
#Tau = np.array(Tau)*1e3
#T10 = np.array(T10)*1e3
#m = next(markers)
#bx.plot(T,T10,m,alpha=0.6,label='T2594 Al')
#ax.plot(T,Tau,m,alpha=0.6,label='T2594 Al')
#
## T2594 Ag
#mask = [x for x in files if ((("TRCL" in x) & (x.endswith(".dat")))& ("T2594"in x) & (not ("Rampe"in x)) & ("Ag" in x))]
#T = list()
#Tau = list()
#T10 = list()
#for p in mask:
#    idx = p.find('K')
#    Temp = int(p[idx-3:idx])
#    A,tau,t10,A1,A2,tau1,tau2,R = process_fromFile(p,save=False,autoclose=True,merge=True)
#    if ((1-R)>1e-2):continue
#    taueff = -(A1*tau1+A2*tau2)/(A1+A2)
#    T.append(Temp)
#    Tau.append(taueff)
#    T10.append(t10)
#
#T = np.array(T)
#Tau = np.array(Tau)*1e3
#T10 = np.array(T10)*1e3
#m = next(markers)
#ax.plot(T,Tau,m,alpha=0.6,label='T2594 Ag')
#bx.plot(T,T10,m,alpha=0.6,label='T2594 Ag')



ax.set_xlabel(r"T (K)")
ax.set_ylabel(r"$\tau$ (ps)")
ax.legend()
#
#bx.set_xlabel("Temperature (K)")
#bx.set_ylabel(r"$\tau_{10}$ (ps)")
#bx.legend()

plt.show()
