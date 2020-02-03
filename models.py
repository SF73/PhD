# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:32:51 2019

@author: sylvain.finot
"""

import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
import scipy.special as sse
from scipy.integrate import quad
import sympy as sy
from sympy.functions import *



def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0):

    errfunc = lambda p, x, y: function(x,p) - y

    # Fit first time
    pfit, perr = optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # 100 random data sets are generated and fitted
    ps = []
    for i in range(100):

        randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            optimize.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)

        ps.append(randomfit) 

    ps = np.array(ps)
    return ps
    #mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
   # Nsigma = 2. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    #err_pfit = Nsigma * np.std(ps,0) 

   #pfit_bootstrap = mean_pfit
   # perr_bootstrap = err_pfit
   # return pfit_bootstrap, perr_bootstrap 





def decay_func(t,t0,A,K):
    return A*np.exp(K*(t-t0))

def double_decay(t,A1,A2,K1,K2,c):
    return A1*np.exp(K1*t)+A2*np.exp(K2*t)+c
def gaussian_heaviside(t,sig,t0):
    return 1/2*(1+sse.erf((t-t0)/(np.sqrt(2)*sig)))
def model_func(t,A,K,sig,t0):
    return gaussian_heaviside(t,sig,t0)*A*np.exp(K*(t-t0))

def model2_func(t,A1,K1,A2,K2,sig,t0):
    return gaussian_heaviside(t,sig,t0)*(A1*np.exp(K1*(t-t0))+A2*np.exp(K2*(t-t0)))
def model2log_func(t,A1,K1,A2,K2,sig,t0):
    return np.log(gaussian_heaviside(t,sig,t0)*(A1*np.exp(K1*(t-t0))+A2*np.exp(K2*(t-t0))))

def model2_ff(t,p):
    return model2_func(t,*p)
def stretched_exp(t,A,K,B):
    return A*np.exp(-np.power((K*(t)),B))#*1/2*(1+sse.erf((t)/(np.sqrt(2)*sig)))

def s_fit(t,y,p0=None):
    popt,pcov = curve_fit(stretched_exp,t,y,p0)#,p0,bounds=(lowerbound,upperbound))
    return popt, pcov
def decay_fit(t,y):
    p0 = [min(t),max(y),1]
    popt, pcov = curve_fit(decay_func, t, y,p0)
    return popt,pcov


def fit(t,y,C=0):
    y = np.abs(y - C)+1e-4
    y = np.log(y)
    t=t-min(t)
    def func(t, A,K,t0):
        return A + K*(t)
    popt, pcov = curve_fit(func, t, y)
    return popt,pcov

def model_fit(t,y,p0):
    A = p0[0]
    K = p0[1]
    sig = p0[2]
    t0 = p0[3]
#    lowerbound = (A*0.001,K*10,0,0)
#    upperbound = (A*1000,K*0.1,np.inf,np.inf)
    popt,pcov = curve_fit(model_func,t,y,p0)#,bounds=(lowerbound,upperbound))
    return popt, pcov

def model2_fit(t,y,p0):
    # A1 = p0[0]
    # K1 = p0[1]
    # A2 = p0[2]
    # K2 = p0[3]
    # sig = p0[4]
    # t0 = p0[5]
#    lowerbound = (A1*0.5,K1*3,-np.inf,-np.inf,0,0)
#    upperbound = (A1*2,K1*0.3,np.inf,+np.inf,1,np.inf)
    
#    lowerbound = (A1*0.5,K1*1.5,-np.inf,-np.inf,0,0)
#    upperbound = (A1*2,K1*0.3,np.inf,+np.inf,1,np.inf)
    
#    lowerbound = (0,-np.inf,-np.inf,-np.inf,0,0)
#    upperbound = (np.inf,0,np.inf,+np.inf,np.inf,np.inf)
#    print(p0)
#    print(lowerbound)
#    print(upperbound)
    popt,pcov = curve_fit(model2_func,t,y,p0)#,bounds=(lowerbound,upperbound))
    return popt, pcov

def model2_fit_bootstrap(t,y,p0):
    ps = fit_bootstrap(p0,t,y,model2_ff)
    A1 = ps[:,0]
    K1 = ps[:,1]
    A2 = ps[:,2]
    K2 = ps[:,3]
    taueff = (A1*(-1/K1)+A2*(-1/K2))/(A1+A2)
    meantaueff = np.mean(taueff)
    dtaueff = np.std(taueff)
    print("BOOTSTRAP")
    print(meantaueff)
    print(dtaueff)
    popt = np.mean(ps,0)
    pcov = np.std(ps,0)
    print("ENDBOOTSTRAP")
    return popt, pcov







s = sy.symbols("s")
t=sy.symbols("t",positive=True)
A = sy.symbols("A")
sig,mu=sy.symbols("sigma,mu",positive=True)
gauss = 1/(sqrt(2*sy.pi)*sig)*exp(-(1/2)*((s-mu)/sig)**2)*Heaviside(s)
f = sy.laplace_transform(gauss,s,t)
gaussian_decay=np.vectorize(sy.lambdify((t,A,sig,mu),A*f[0],modules=['numpy']))

#logN = 1/(sqrt(2*sy.pi)*sig*s)*exp(-(1/2)*(log(s-mu)/sig)**2)
#f = sy.laplace_transform(logN,s,t)
#lognormal_decay=np.vectorize(sy.lambdify((t,A,sig,mu),A*f[0],modules=['numpy']))

def gaussian_fit(t,y,p0,baseline):
    popt,pcov = curve_fit(gaussian_decay,t,y,p0)
    return popt,pcov


def DLT(t,A,gamma,t0):
    """Discrete Laplace Transform"""
    y = 0
    for i in range(len(A)):
        y += A[i] * np.exp(-gamma[i]*(t-t0))
    y = y/np.sum(A)
    return y

def LT(rho):
    def integrand(gam,t,t0):
        return rho(gam)*np.exp(-gam*(t-t0))
    def function(t,t0):
        return quad(integrand,0,np.inf,args=(t,t0))[0]
    return np.vectorize(function)
#def init_decay():
#    s = sy.symbols("s")
#    t=sy.symbols("t")
#    sig,mu=sy.symbols("sigma,mu",positive=True)
#    Lap = 1/(sqrt(2*sy.pi)*sig)*exp(-(1/2)*((s-mu)/sig)**2)*Heaviside(s)
#    f = sy.laplace_transform(Lap,s,t)
#    Intensity=sy.lambdify((t,sig,mu),f[0],modules=['sympy'])
##    TD = sy.lambdify((s,mu,sig),Lap,modules=['sympy'])
##    time = np.linspace(0,7,500)
##    freq = np.linspace(0.1,20,251)
##    vTD = np.vectorize(TD)
#    global gaussian_decay
#    gaussian_decay = np.vectorize(Intensity)
