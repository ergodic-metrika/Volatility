# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:01:21 2022

@author: sigma
"""

# Data Manipulation
import pandas as pd
from numpy import *
from datetime import timedelta
import yfinance as yf
from tabulate import tabulate

# Math & Optimization
from scipy.stats import norm
from scipy.optimize import fsolve

# Plotting
import matplotlib.pyplot as plt
import cufflinks as cf
cf.set_config_file(offline=True)

from BS import BS
from Bisection import bisection

#Newton method
def newton_iv(className, spot, strike, rate, dte, volatility, callprice=None, putprice=None):
    
    x0 = 1 # initial guess
    h = 0.001
    tolerance = 1e-7
    epsilon = 1e-14 # some kind of error or floor
    
    maxiter = 200
    
    if callprice:
        # f(x) = Black Scholes Call price - Market Price
        f = lambda x: eval(className)(spot, strike, rate, dte, x).callPrice - callprice
    if putprice:
        f = lambda x: eval(className)(spot, strike, rate, dte, x).putPrice - putprice
        
    for i in range(maxiter):
        y = f(x0)
        yprime = (f(x0+h) - f(x0-h))/(2*h) # central difference
        
        if abs(yprime)<epsilon:
            break # this is critial, because volatility cannot be negative
        x1 = x0 - y/yprime
        
        if (abs(x1-x0) <= tolerance*abs(x1)):
            break
        x0=x1
        
    return x1


opt = BS(14637,14600,0.0087,16/250,0.18)

opt.callPrice
opt.putPrice

newton_iv('BS',14637,14600,0.0087,16/250,0.18,callprice=262)

newton_iv('BS',14637,14600,0.0087,16/250,0.18,putprice=230)


# Bisection Method
def bisection_iv(className, spot, strike, rate, dte, volatility, callprice=None, putprice=None, high=500.0, low=0.0):
    
    if callprice:
        price = callprice
    if putprice and not callprice:
        price = putprice
        
    tolerance = 1e-7
        
    for i in range(10000):
        mid = (high + low) / 2 # c= (a+b)/2
        if mid < tolerance:
            mid = tolerance
            
        if callprice:
            estimate = eval(className)(spot, strike, rate, dte, mid).callPrice # Blackscholes price
        if putprice:
            estimate = eval(className)(spot, strike, rate, dte, mid).putPrice
        
        if round(estimate,6) == price:
            break
        elif estimate > price: 
            high = mid # b = c
        elif estimate < price: 
            low = mid # a = c
    
    return mid



bisection_iv('BS',14648,14600,0.0087,2/250,0.18,callprice=127)
             
bisection_iv('BS',14648,14600,0.0087,2/250,0.18,putprice=52)
