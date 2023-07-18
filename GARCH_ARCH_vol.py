# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:08:42 2022

@author: sigma
"""

# Import required libraries
import pandas as pd
import numpy as np

# Import yahoo finance libraries

#pip install pandas_datareader
import pandas_datareader as pdr
import yfinance as yf

# Import cufflinks for visualization
import cufflinks as cf
cf.set_config_file(offline=True)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


from scipy.stats import norm
from scipy.optimize import minimize

# Import matplotlib for visualization
import matplotlib
import matplotlib.pyplot as plt

# Plot settings
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['lines.linewidth'] = 2.0


# Fetch data by specifying the the start and end dates
df = yf.download('ETH-USD', start='2015-08-07', end='2022-11-08', progress=False)

# Display the first five rows of the dataframe to check the results. 
df.head()


# Calculate daily returns
returns = np.log(df['Adj Close']).diff().fillna(0)

np.var(returns)


# Visualize ETH daily returns
plt.plot(returns, color='orange')
plt.title('ETH Returns')
plt.grid(True)


# GARCH(1,1) function
def garch(omega, alpha, beta, ret):
    
    length = len(ret)
    
    var = []
    for i in range(length):
        if i==0:
            var.append(omega/np.abs(1-alpha-beta))
        else:
            var.append(omega + alpha * ret[i-1]**2 + beta * var[i-1])
            
    return np.array(var)

# Log likelihood function
def likelihood(params, ret):
    
    length = len(ret)
    omega = params[0]
    alpha = params[1]
    beta = params[2]
    
    variance = garch(omega, alpha, beta, ret)
    
    llh = []
    for i in range(length):
        llh.append(np.log(norm.pdf(ret[i], 0, np.sqrt(variance[i]))))
    
    return -np.sum(np.array(llh))

# Specify optimization input
param = ['omega', 'alpha', 'beta']
initial_values = (np.var(returns), 0.1,0.8)

res = minimize(likelihood, initial_values, args = returns, 
                   method='Nelder-Mead', options={'disp':False})
res


res['x']

# GARCH parameters
dict(zip(param,np.around(res['x']*100,4)))

# Parameters
omega = res['x'][0] 
alpha = res['x'][1]
beta = res['x'][2]

# Variance
var = garch(res['x'][0],res['x'][1],res['x'][2],returns)

# Annualised conditional volatility
ann_vol = np.sqrt(var*252) * 100
ann_vol

# Visualise GARCH volatility and VIX
plt.title('Annualized Volatility')
plt.plot(returns.index, ann_vol, color='orange', label='GARCH')

# Calculate N-day forecast
longrun_variance = omega/(1-alpha-beta)
 
fvar = []
for i in range(1,732):    
    fvar.append(longrun_variance + (alpha+beta)**i * (var[-1] - longrun_variance))

var = np.array(fvar)

# Verify first 10 values
var[:10]

# Plot volatility forecast over different time horizon
plt.axhline(y=np.sqrt(longrun_variance*252)*100, color='blue')
plt.plot(np.sqrt(var*252)*100, color='red')

plt.xlabel('Horizon (in days)')
plt.ylabel('Volatility (%)')

plt.annotate('GARCH Forecast', xy=(650,80), color='red')
plt.annotate('Longrun Volatility =' + str(np.around(np.sqrt(longrun_variance*252)*100,2)) + '%', 
             xy=(0,86), color='blue')

plt.title('Volatility Forecast : N-days Ahead')
plt.grid(axis='x')

#ARCH toolbox
#pip install arch
# Import arch library
from arch import arch_model

# Mean zero
g1 = arch_model(returns, vol='GARCH', mean='Zero', p=1, o=0, q=1, dist='Normal')

model = g1.fit()

# Model output
print(model)

# Model params
model.params

# Plot annualised vol
fig = model.plot(annualize='D')

model.conditional_volatility*np.sqrt(252)

# Constant mean
g2 = arch_model(returns, vol='GARCH', mean='Constant', p=1, o=0, q=1, dist='Normal')

# Model output
model2 = g2.fit(disp='off')
print(model2)

# Forecast for next 60 days
model_forecast = model.forecast(horizon=60)

# Subsume forecast values into a dataframe
forecast_df = pd.DataFrame(np.sqrt(model_forecast.variance.dropna().T *252)*100)
forecast_df.columns = ['Cond_Vol']
forecast_df.head()

# long run variance from model forecast
lrv = model.params[0]/(1-model.params[1]-model.params[2])

# long run variance
np.sqrt(lrv*252)*100

# Plot volatility forecast over a 60-day horizon
plt.plot(forecast_df, color='blue')
plt.xlim(0,60)
plt.xticks(rotation=90)
plt.xlabel('Horizon (in days)')
plt.ylabel('Volatility (%)')
plt.title('Volatility Forecast : 60-days Ahead');
plt.grid(True)
