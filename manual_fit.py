# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:02:43 2019

@author: sylvain.finot
"""

#manual fit

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
from batchProcessing import getListOfFiles


class manual_Fitter():
    def __init__(self,path):
        self.path=path
        counts = np.loadtxt(path) #Full histogram
        binNumber = int(counts[0]) #Nombre de bin
        self.binsize = counts[3] #Taille d'un bin
        self.counts = counts[-binNumber:] #histogramh
        self.t = np.arange(binNumber)*self.binsize #echelle de temps en ns
        self.shift_is_held = False
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_alpha(0)
        self.ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        self.ax.set_xlabel("t (ns)")
        self.ax.set_ylabel("Intensity (arb. unit)")
        self.line = self.ax.plot(self.t,self.counts,'.')[0]
        self.span = SpanSelector(self.ax, lambda xmin,xmax:self.onselect(self.line,xmin,xmax), 'horizontal', useblit=True,
                rectprops=dict(alpha=0.5, facecolor='red'))
        self.fig.canvas.mpl_connect('key_press_event',lambda event: self.on_key_press(event))
        self.fig.canvas.mpl_connect('key_release_event',lambda event: self.on_key_release(event))
        self.baseline = 0
        self.fit_time = []
        self.fit_counts= []
        
    def on_key_press(self, event):
            if event.key == 'shift':
               self.shift_is_held = True
    def on_key_release(self, event):
        if event.key == 'shift':
           self.shift_is_held = False
    def process(self):
        leftl = self.fit_counts.argmax()
        rightl = np.nan
        c=10e-2#np.exp(-1)


        while(rightl<leftl or np.isnan(rightl)):        
            threshold = (max(self.fit_counts)-self.baseline)*c+self.baseline #(max(reduced_counts)-baseline)*np.exp(-3)
            mask = np.convolve(np.sign(self.fit_counts-threshold),[-1,1],'same') #detect le chgt de sign de reduced-threshold
            mask[0] = 0
            rightl = np.argmax(mask)
            c += 0.005
        t0 = self.fit_time[leftl]
        t10 = self.fit_time[rightl] - t0
        print("t10 : %.4e"%t10)
        SNR = max(self.fit_counts)/self.baseline
        
        print("SNR : %.4f"%SNR)
        
        
        
        popt,pcov= fit(self.fit_time[leftl:rightl],self.fit_counts[leftl:rightl],self.baseline)
        A_lin=np.exp(popt[0])
        K_lin = popt[1]
        p_sigma = np.sqrt(np.diag(pcov))
        R = R2(self.fit_counts,decay_func(self.fit_time,t0,A_lin,K_lin)+self.baseline)
    #    ax.plot(fit_time,decay_func(fit_time,t0,A_lin,K_lin)+baseline,label=r'decay fit $R^2 =$%.4f%s$\tau_{eff} =$ %.2e ns'%(R,'\n',-1/K_lin))
        print(R)
        
        
    #    Stretched
    #    popt,pcov = s_fit(fit_time-t0,fit_count-baseline,[A_lin,abs(K_lin),1])
    #    print(popt)
    #    ax.plot(reduced_time,stretched_exp(reduced_time-t0,*popt)+baseline,c='orange',label=r'stretched')
        
        
        #calcul de la limite de gauche
#        c=np.exp(-3)
#        while(np.isnan(leftl)):        
#            threshold = (max(reduced_counts)-leftBaseline)*c #(max(reduced_counts)-baseline)*np.exp(-3)
#            mask = np.convolve(np.sign(reduced_counts-threshold),[1,-1],'same') #detect le chgt de sign de reduced-threshold
#            mask[0] = 0
#            leftl = np.argmax(mask)
#            print(leftl)
#            c += 0.01
        print("------------------model1-----------------------")
        #fit simple exp convoluée avec heaviside
    #    if not baselineError:
#        fit_time = reduced_time[leftl-50:]
#        fit_count = reduced_counts[leftl-50:]
        init = [A_lin,K_lin,0.02,t0]
        popt,pcov= model_fit(self.fit_time,self.fit_counts-self.baseline,init)
        print(popt)
        A=popt[0]
        K = popt[1]
        sig = popt[2]
        t0 = popt[3]
        R = R2(self.fit_counts,model_func(self.fit_time,*popt)+self.baseline)
        Radj = R2adj(len(self.fit_counts),3,self.fit_counts,model_func(self.fit_time,*popt)+self.baseline)
        rChi = rChi2(len(self.fit_counts),3,self.fit_counts,model_func(self.fit_time,*popt)+self.baseline)
        print("R2 : %.4f"%R)
        print("R2adj : %.4f"%Radj)
        print("rChi2 : %.4f"%rChi)
        
        if not hasattr(self,"simple_line"):
            self.simple_line,= self.ax.plot(self.fit_time,model_func(self.fit_time,*popt)+self.baseline,c='orange',label=r'simple_fit $1-R^2 =$ %.2e %s $\tau_{eff} =$ %.2e ns %s  $\sigma=$%.2e ns'%((1-R),'\n',-1/K,'\n',sig))
        else:
            self.simple_line.set_ydata(model_func(self.fit_time,*popt)+self.baseline)
            self.simple_line.set_xdata(self.fit_time)
            self.simple_line.set_label(r'simple_fit $1-R^2 =$ %.2e %s $\tau_{eff} =$ %.2e ns %s  $\sigma=$%.2e ns'%((1-R),'\n',-1/K,'\n',sig))
        
        
        
        print("------------------model2-------------------")
        init = [A,K,1,1,sig,t0]
        popt,pcov= model2_fit(self.fit_time,self.fit_counts-self.baseline,init)
        print(popt)
        A1 = popt[0]
        K1 = popt[1]
        A2 = popt[2]
        K2 = popt[3]
        sig = popt[4]
        t0 = popt[5]
        R = R2(self.fit_counts,model2_func(self.fit_time,*popt)+self.baseline)
        Radj = R2adj(len(self.fit_counts),3,self.fit_counts,model2_func(self.fit_time,*popt)+self.baseline)
        rChi = rChi2(len(self.fit_counts),3,self.fit_counts,model2_func(self.fit_time,*popt)+self.baseline)
        print("R2 : %.4f"%R)
        print("R2adj : %.4f"%Radj)
        print("rChi2 : %.4f"%rChi)
        print("A1/A2 : %.4e"%(A1/A2))
        if R>0.8:
            taueff = (A1*(-1/K1)+A2*(-1/K2))/(A1+A2)
            print(taueff)
            if not hasattr(self,"double_line"):
                self.double_line,= self.ax.plot(self.fit_time,model2_func(self.fit_time,*popt)+self.baseline,c='red',label=r'double_fit $1-R^2 =$ %.2e %s$A_{1} =$ %.2e %s$A_{2} =$ %.2e %s$\tau_{1} =$ %.2e ns %s$\tau_{2} =$ %.2e ns %s$\sigma =$ %.2e ns %s$\tau_{eff} =$ %.2e ns'%((1-R),'\n',A1,'\n',A2,'\n',-1/K1,'\n',-1/K2,'\n',sig,'\n',taueff))
            else:
                self.double_line.set_ydata(model2_func(self.fit_time,*popt)+self.baseline)
                self.double_line.set_xdata(self.fit_time)
                self.double_line.set_label(r'double_fit $1-R^2 =$ %.2e %s$A_{1} =$ %.2e %s$A_{2} =$ %.2e %s$\tau_{1} =$ %.2e ns %s$\tau_{2} =$ %.2e ns %s$\sigma =$ %.2e ns %s$\tau_{eff} =$ %.2e ns'%((1-R),'\n',A1,'\n',A2,'\n',-1/K1,'\n',-1/K2,'\n',sig,'\n',taueff))
        
        self.ax.legend()
        self.fig.tight_layout()
#        if save==True:  
#            fig.savefig('%s_double_decay.pdf'%os.path.splitext(path)[0])
#            fig.savefig('%s_double_decay.png'%os.path.splitext(path)[0])
#            ax.set_yscale('log')
#            fig.savefig('%s_double_decay_log.pdf'%os.path.splitext(path)[0])
#            fig.savefig('%s_double_decay_log.png'%os.path.splitext(path)[0])
#        if autoclose==True:
#            plt.close(fig)
#        ax.set_yscale('log')
        return A,1/K,A1,A2,1/K1,1/K2,R
    def onselect(self,line,xmin, xmax):
        t=line.get_xdata()
        counts = line.get_ydata()
        ax=line.axes
        fig = line.figure
        indmin, indmax = np.searchsorted(t, (xmin, xmax))
        indmax = min(len(t) - 1, indmax)
    
        thisx = t[indmin:indmax]
        thisy = counts[indmin:indmax]
        if self.shift_is_held:
            self.baseline = np.median(thisy)
        else:
            self.fit_time = thisx
            self.fit_counts=thisy
        if len(self.fit_counts)>1:
            self.process()
            

#        fig.canvas.draw()


#path=r"C:\Users\sylvain.finot\Documents\data\2019-03-22 - T2594 - Rampe\010K\TRCL.dat"
#    path=r'C:/Users/sylvain.finot/Documents/data/2019-03-08 - T2597 - 300K/Fil2/TRCL-L12-5um-cw440nm/TRCL.dat'
#    path = r"C:\Users\sylvain.finot\Documents\data\2019-03-21 - T2594 - 005K\Fil 3\TRCL 4_-04.42um"

# Set useblit=True on most backends for enhanced performance.

    
#    countmax = counts[int(1/binsize):binNumber-int(5/binsize)].argsort()[-1]+int(1/binsize)
#    tmin = max(0,countmax-int(1/binsize))
#    tmax = min(countmax+int(5/binsize),binNumber)
#    reduced_time = t[tmin:tmax]
#    reduced_time = reduced_time - min(reduced_time)
#    if merge=='auto':
#        merge = counts.max() < 5E3
#    if merge==True:
#        reduced_counts = mergeData(counts,binNumber,binsize,name)
#    else:
#        reduced_counts = counts[tmin:tmax]
#    rightBaseline = np.median(reduced_counts[-int(1/binsize):])
#    leftBaseline  = np.median(reduced_counts[:int(1/binsize)])
#    baselineError = abs(rightBaseline-leftBaseline)/rightBaseline > 0.50
#    ax.plot(reduced_time,reduced_counts,'.',c='k',label='data')
#    baseline = rightBaseline
#    
#    #calcul de la limite de droite pour fit
#    leftl = reduced_counts.argmax()+2
#    rightl = 0
#    c=np.exp(-3)
#    while(rightl<leftl or np.isnan(rightl)):        
#        threshold = (max(reduced_counts)-baseline)*c #(max(reduced_counts)-baseline)*np.exp(-3)
#        mask = np.convolve(np.sign(reduced_counts-threshold),[-1,1],'same') #detect le chgt de sign de reduced-threshold
#        mask[0] = 0
#        rightl = np.argmax(mask)
#        print(rightl)
#        c += 0.01
#        
#    t0 = reduced_time[leftl]#temps correspondant au max
#    
#    
#    #Fit exponential decay
#    print("simple decay")
#    #seuleement decay (pas de montée)
#    fit_time = reduced_time[leftl:rightl]
#    fit_count = reduced_counts[leftl:rightl]
#    
#    popt,pcov= fit(fit_time,fit_count,baseline)
#    A_lin=np.exp(popt[0])
#    K_lin = popt[1]
#    p_sigma = np.sqrt(np.diag(pcov))
#    R = R2(fit_time-t0,fit_count,decay_func(fit_time,t0,A_lin,K_lin)+baseline)
##    ax.plot(fit_time,decay_func(fit_time,t0,A_lin,K_lin)+baseline,label=r'decay fit $R^2 =$%.4f%s$\tau_{eff} =$ %.2e ns'%(R,'\n',-1/K_lin))
#    print(R)
#    
#    
##    Stretched
##    popt,pcov = s_fit(fit_time-t0,fit_count-baseline,[A_lin,abs(K_lin),1])
##    print(popt)
##    ax.plot(reduced_time,stretched_exp(reduced_time-t0,*popt)+baseline,c='orange',label=r'stretched')
#    
#    
#    #calcul de la limite de gauche
#    c=np.exp(-3)
#    while(np.isnan(leftl)):        
#        threshold = (max(reduced_counts)-leftBaseline)*c #(max(reduced_counts)-baseline)*np.exp(-3)
#        mask = np.convolve(np.sign(reduced_counts-threshold),[1,-1],'same') #detect le chgt de sign de reduced-threshold
#        mask[0] = 0
#        leftl = np.argmax(mask)
#        print(leftl)
#        c += 0.01
#    print("------------------model1-----------------------")
#    #fit simple exp convoluée avec heaviside
##    if not baselineError:
#    fit_time = reduced_time[leftl-50:]
#    fit_count = reduced_counts[leftl-50:]
#    init = [A_lin,K_lin,0.02,t0]
#    popt,pcov= model_fit(fit_time,fit_count-baseline,init)
#    print(popt)
#    A=popt[0]
#    K = popt[1]
#    sig = popt[2]
#    t0 = popt[3]
#    R = R2(fit_time,fit_count,model_func(fit_time,*popt)+baseline)
#    print(R)
#    
#    ax.plot(fit_time,model_func(fit_time,*popt)+baseline,c='orange',label=r'simple_fit $1-R^2 =$ %.2e %s $\tau_{eff} =$ %.2e ns %s  $\sigma=$%.2e ns'%((1-R),'\n',-1/K,'\n',sig))
#
#    
#    
#    
#    print("------------------model2-------------------")
#    init = [A,K,1,1,sig,t0]
#    popt,pcov= model2_fit(fit_time,fit_count-baseline,init)
#    print(popt)
#    A1 = popt[0]
#    K1 = popt[1]
#    A2 = popt[2]
#    K2 = popt[3]
#    sig = popt[4]
#    t0 = popt[5]
#    R = R2(fit_time,fit_count,model2_func(fit_time,*popt)+baseline)
#    print(R)
#    if R>0.8:
#        taueff = (A1*(-1/K1)+A2*(-1/K2))/(A1+A2)
#        print(taueff)
#    
#        ax.plot(fit_time,model2_func(fit_time,*popt)+baseline,c='red',label=r'double_fit $1-R^2 =$ %.2e %s$A_{1} =$ %.2e %s$A_{2} =$ %.2e %s$\tau_{1} =$ %.2e ns %s$\tau_{2} =$ %.2e ns %s$\sigma =$ %.2e ns %s$\tau_{eff} =$ %.2e ns'%((1-R),'\n',A1,'\n',A2,'\n',-1/K1,'\n',-1/K2,'\n',sig,'\n',taueff))
#    
#    ax.legend()
#    fig.tight_layout()
#    if save==True:  
#        fig.savefig('%s_double_decay.pdf'%os.path.splitext(path)[0])
#        fig.savefig('%s_double_decay.png'%os.path.splitext(path)[0])
#        ax.set_yscale('log')
#        fig.savefig('%s_double_decay_log.pdf'%os.path.splitext(path)[0])
#        fig.savefig('%s_double_decay_log.png'%os.path.splitext(path)[0])
#    if autoclose==True:
#        plt.close(fig)
#    ax.set_yscale('log')
#    return A,1/K,A1,A2,1/K1,1/K2,R
#def process_all():
#    files=getListOfFiles(r'C:/Users/sylvain.finot/Documents/data/')
#    mask = [x for x in files if (("TRCL" in x) & (x.endswith(".dat")))]
#    for p in mask:
#        try:
#            process(p,autoclose=True)
#        except:
#            pass
        

#def main():
#    path = input("Enter the path of your file: ")
#    path=path.replace('"','')
#    path=path.replace("'",'')
##    path = r'C:/Users/sylvain.finot/Documents/data/2019-03-11 - T2597 - 5K/Fil3/TRCL-cw455nm/TRCL.dat'
#    process(path,save=False,autoclose=False,merge=True)
    
#if __name__ == '__main__':
#    main()