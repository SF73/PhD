# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:32:51 2019

@author: sylvain.finot
"""

import numpy as np
from scipy.optimize import curve_fit
import scipy.special as sse
from scipy.integrate import quad
import sympy as sy
from sympy.functions import *

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
    A1 = p0[0]
    K1 = p0[1]
    A2 = p0[2]
    K2 = p0[3]
    sig = p0[4]
    t0 = p0[5]
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
