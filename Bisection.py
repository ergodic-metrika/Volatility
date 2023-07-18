# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:21:37 2022

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