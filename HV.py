# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:18:24 2022

@author: sigma
"""

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd

import scipy.stats

import statsmodels.api as sm

import pandas_datareader as pdr
import yfinance as yf



# Fetch data by specifying the the start and end dates
#Note ^HSI: Hang Seng Index
df = yf.download('^HSI', start='2007-01-03', end='2022-11-09', progress=False)

# Display the first five rows of the dataframe to check the results. 
df.head()

#df=pd.read_excel(r'D:\Derivatives Trading\ResearchRecord.xlsx')


#timeseries1 ={'Date':df["Date"], 'Spot':df["Spot"], 'IV':df["Implied Volatility"], 'pnl': df["PNL Index"]}


daily_return=df["Adj Close"].pct_change()


#calculate 20-day historical volatility
rolling_sd_return_20=daily_return.rolling(20).std()
rolling_sd_return_20

volatility_20_day=rolling_sd_return_20*(250**0.5)
volatility_20_day
list(volatility_20_day)
plt.plot(volatility_20_day, color='indigo')
plt.title('20D HV')

#calculate 5-day historical volatility
rolling_sd_return_5=daily_return.rolling(5).std()
rolling_sd_return_5

volatility_5_day=rolling_sd_return_5*(250**0.5)
volatility_5_day
list(volatility_5_day)
plt.plot(volatility_5_day, color='violet')
plt.title('5D HV')
