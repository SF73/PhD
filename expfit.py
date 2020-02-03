# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:50:23 2019

@author: sylvain.finot
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from mergeNoise import mergeData
from stats import R2,R2adj,rChi2
from models import *
from FileHelper import getListOfFiles
#from batchProcessing import getListOfFiles
#plt.rc('font', family='serif',size=12)
#plt.rc('text', usetex=False)
#plt.rc('xtick', labelsize=14)
#plt.rc('ytick', labelsize=14)
#plt.rc('axes', labelsize=18)
plt.style.use('Rapport')

def process_fromFile(path,save=False,autoclose=False,merge=False,fig=None):
#    path = r"F:\data\2019-03-08 - T2455 - withUL\TRCL-440nm.dat"
    name = path[path.find('2019'):]
    print('path:')
    print(path)
    if autoclose:
        plt.ioff()
    else:
        plt.ion()
    
    if len(name)>100:name=''
    if fig is None:
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        ax.set_xlabel("t (ns)")
        ax.set_ylabel("Intensity (arb. unit)")
        ax.set_title("Carrier lifetime : %s"%name)
    else:
        ax = fig.gca()
    
    counts = np.loadtxt(path) #Full histogram
    binNumber = int(counts[0]) #Nombre de bin
    binsize = counts[3] #Taille d'un bin
    counts = counts[-binNumber:] #histogram
#    counts = counts/max(counts)
    t = np.arange(binNumber)*binsize #echelle de temps en ns

    #distance la plus courte si on travaille a 100MHz -> 200MHz (montee + descente) -> periode de 5ns
    #detection si peak est grand de 0.1 max
    peaks,_ = scipy.signal.find_peaks(counts,height=0.1*max(counts),distance=5/binsize)
    p1 = np.convolve(peaks,[1,-1],mode='same')[1::2].mean()*binsize
    p2 = np.convolve(peaks,[1,-1],mode='same')[2::2].mean()*binsize
    meanfreq = 1e-6/((p1+p2)*1e-9)
    print("Mean freq : %2.2f"%meanfreq)
    print("Short period : %2.2f ns"%max(p1,p2))
    print("Long period : %2.2f ns"%min(p1,p2))
#    test,testa = plt.subplots()
#    testa.plot(t,counts)
#    asymetrie = max(p1,p2)/(p1+p2) - 0.5
    countmax = counts[int(1/binsize):binNumber-int(5/binsize)].argsort()[-1]#+int(1/binsize)
#    maxs = np.arange(-5,5)*1e9/efreq + counts[int(1/binsize):binNumber-int(5/binsize)].argmax()*binsize
#    maxs = maxs[(maxs>=0) & (maxs<t[-1])]
#    axes[0].plot(maxs,1e4*np.ones(len(maxs)),'D',c='r')
    tmin = max(0,countmax-int(1/binsize))
    tmax = min(countmax+int(5/binsize),binNumber)
    if merge=='auto':
        merge = counts.max() < 1E4
    if merge==True:
        reduced_time,reduced_counts = mergeData(t,counts,binsize,name,show=False)
    else:
        reduced_time = t[tmin:tmax]
        reduced_counts = counts[tmin:tmax]
    reduced_time = reduced_time - min(reduced_time)
    rightBaseline = np.median(reduced_counts[-int(1/binsize):])
    leftBaseline  = np.median(reduced_counts[:int(1/binsize)])
    baselineError = abs(rightBaseline-leftBaseline)/min(rightBaseline,leftBaseline) > 0.20
    print("Asymetric baseline : %s" %(str(baselineError)))
    baseline = rightBaseline
#    if baselineError:
#        reduced_time = t[tmin:tmax]
#        reduced_counts = counts[tmin:tmax]
#        reduced_time = reduced_time - min(reduced_time)
#        rightBaseline = np.median(reduced_counts[-int(1/binsize):])
#        leftBaseline  = np.median(reduced_counts[:int(1/binsize)])
#        baseline = rightBaseline
##    ax.plot(reduced_time,reduced_counts,'.',c='k',label='data')
#        baseline = min(rightBaseline,leftBaseline)
    #calcul de la limite de droite pour fit
    leftl = reduced_counts.argmax()+2
    rightl = np.nan
    c=10e-2#np.exp(-1)
    while(np.isnan(rightl) or rightl<leftl):        
        threshold = (max(reduced_counts)-baseline)*c+baseline #(max(reduced_counts)-baseline)*np.exp(-3)
        mask = np.convolve(np.sign(reduced_counts-threshold),[-1,1],'same') #detect le chgt de sign de reduced-threshold
        print(mask)
        mask[0] = 0
        rightl = np.argmax(mask)
        c += 0.005
    c -= 0.005
    t0 = reduced_time[leftl]#temps correspondant au max
    t10 = reduced_time[rightl] - t0
    ax.plot(reduced_time,reduced_counts,'.',c='k',label=r'data | $\tau_{%.2d}$ = %.2e'%(np.round((c-0.005)*100),t10))
    SNR = max(reduced_counts)/baseline
    print("tau %.4f : %.4f"%(t10,c))
    print("SNR : %.4f"%SNR)
    #Fit exponential decay
    print("------------------simple decay------------------")
    #seuleement decay (pas de montée)
    fit_time = reduced_time[leftl:rightl]
    fit_count = reduced_counts[leftl:rightl]
    
    popt,pcov= fit(fit_time,fit_count,baseline)
    A_lin=np.exp(popt[0])
    K_lin = popt[1]
    p_sigma = np.sqrt(np.diag(pcov))
    R = R2(fit_count,decay_func(fit_time,t0,A_lin,K_lin)+baseline)
    Radj = R2adj(len(fit_count),2,fit_count,decay_func(fit_time,t0,A_lin,K_lin)+baseline)
    rChi = rChi2(len(fit_count),2,fit_count,decay_func(fit_time,t0,A_lin,K_lin)+baseline)
#    ax.plot(fit_time,decay_func(fit_time,t0,A_lin,K_lin)+baseline,label=r'decay fit $R^2 =$%.4f%s$\tau_{eff} =$ %.2e ns'%(R,'\n',-1/K_lin))
    print("R2 : %.4f"%R)
    print("R2adj : %.4f"%Radj)
    print("rChi2 : %.4f"%rChi)
    
    fit_time = reduced_time[leftl:-300]
    fit_count = reduced_counts[leftl:-300]
    
    #calcul de la limite de gauche
    c=10e-2
    leftl = np.nan
    while(np.isnan(leftl)):        
        threshold = (max(reduced_counts)-leftBaseline)*c+leftBaseline #(max(reduced_counts)-baseline)*np.exp(-3)
        print(threshold)
        mask = np.convolve(np.sign(reduced_counts-threshold),[1,-1],'same') #detect le chgt de sign de reduced-threshold
        mask[0] = 0
        leftl = np.argmax(mask)
        c += 0.01
    
#    if not baselineError:
    fit_time = reduced_time[leftl:]
    fit_count = reduced_counts[leftl:]
    try:
        print("------------------model1-----------------------")
        #fit simple exp convoluée avec heaviside
        init = [A_lin,K_lin,0.02,t0]
        popt,pcov= model_fit(fit_time,fit_count-baseline,init)
        print(popt)
        print(pcov)
        A,K,sig,t0=popt
        tauerror = np.sqrt(pcov[1,1])
        Rmono = R2(fit_count,model_func(fit_time,*popt)+baseline)
        Radj = R2adj(len(fit_count),3,fit_count,model_func(fit_time,*popt)+baseline)
        rChi = rChi2(len(fit_count),3,fit_count,model_func(fit_time,*popt)+baseline)
    #    ax.plot(fit_time,decay_func(fit_time,t0,A_lin,K_lin)+baseline,label=r'decay fit $R^2 =$%.4f%s$\tau_{eff} =$ %.2e ns'%(R,'\n',-1/K_lin))
        print("R2 : %.4f"%Rmono)
        print("R2adj : %.4f"%Radj)
        print("rChi2 : %.4f"%rChi)
        print("tau : %.4f +- %.4f"%(-1/K,tauerror))
        ax.plot(fit_time,model_func(fit_time,*popt)+baseline,c='orange',label=r'simple_fit $1-R^2 =$ %.2e %s $\tau_{eff} =$ %.2e ns %s  $\sigma=$%.2e ns'%((1-Rmono),'\n',-1/K,'\n',sig))
    
        
        
        print("------------------model2-----------------------")
        init = [A/2,K,A/2,K,sig,t0]
        #popt,pcov = model2_fit_bootstrap(fit_time,fit_count-baseline,init)
        popt,pcov= model2_fit(fit_time,fit_count-baseline,init)
        #print(popt)
        #print(pcov)
        #print(np.sqrt(np.diag(pcov)))
        A1,K1,A2,K2,sig,t0 = popt
        dA2 = A1*(K1-K2)/(K1*K2*(A1+A2)**2)
        dA1 = A2*(K2-K1)/(K1*K2*(A1+A2)**2)
        dK1 = -A1/((K1**2)*(A1+A2)**2)
        dK2 = -A2/((K2**2)*(A1+A2)**2)
        VA1, VK1, VA2, VK2, Vsig, Vt0 = np.diag(pcov)
        tauefferror = np.sqrt((dA1**2)*VA1+(dA2**2)*VA2+(dK1**2)*VK1+(dK2**2)*VK2)
        Rbi = R2(fit_count,model2_func(fit_time,*popt)+baseline)
        Radj = R2adj(len(fit_count),3,fit_count,model2_func(fit_time,*popt)+baseline)
        rChi = rChi2(len(fit_count),3,fit_count,model2_func(fit_time,*popt)+baseline)
        print("R2 : %.4f"%R)
        print("R2adj : %.4f"%Radj)
        print("rChi2 : %.4f"%rChi)
        print("A1/A2 : %.4e"%(A1/A2))
        taueff = (A1*(-1/K1)+A2*(-1/K2))/(A1+A2)
        
        print("taueff : %.4e ns +- %.4e ns"%(taueff,tauefferror))
        if Rbi>0.8:
            ax.plot(fit_time,model2_func(fit_time,*popt)+baseline,c='red',label=r'double_fit $1-R^2 =$ %.2e %s$A_{1} =$ %.2e %s$A_{2} =$ %.2e %s$\tau_{1} =$ %.2e ns %s$\tau_{2} =$ %.2e ns %s$\sigma =$ %.2e ns %s$\tau_{eff} =$ %.2e ns'%((1-Rbi),'\n',A1,'\n',A2,'\n',-1/K1,'\n',-1/K2,'\n',sig,'\n',taueff))
        
        ax.legend()
#        fig.tight_layout()
        if save==True:  
            fig.savefig('%s_double_decay.pdf'%os.path.splitext(path)[0])
            fig.savefig('%s_double_decay.png'%os.path.splitext(path)[0])
            ax.set_yscale('log')
            fig.savefig('%s_double_decay_log.pdf'%os.path.splitext(path)[0])
            fig.savefig('%s_double_decay_log.png'%os.path.splitext(path)[0])
        if autoclose==True:
            plt.close(fig)
        ax.set_yscale('log')
        return A,-1/K,t10,A1,A2,1/K1,1/K2,Rmono,tauerror,Rbi,tauefferror
    except Exception as e:
        print(e)
        return None
def process_all(path):
    files=getListOfFiles(path)
    mask = [x for x in files if (("TRCL" in x) & (x.endswith(".dat")))]
    for p in mask:
        try:
            process_fromFile(p,save=True,autoclose=False)
        except:
            pass

def main():
    path = input("Enter the path of your file: ")
    path=path.replace('"','')
    path=path.replace("'",'')
#    path = r'C:/Users/sylvain.finot/Documents/data/2019-03-11 - T2597 - 5K/Fil3/TRCL-cw455nm/TRCL.dat'
    process_fromFile(path,save=False,autoclose=False,merge=True)
    
if __name__ == '__main__':
    main()