# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:58:56 2022

@author: sigma
"""

# Base Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV

# SVM
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error, accuracy_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Create a variable to predict for 'n' days
n = 5

# Load locally stored data

import yfinance as yf
asset=yf.Ticker("^VIX")
df = asset.history(period="3y")
df['Close'].plot(title="VIX trend")
df['Close']



df['Target'] = df['Close'].shift(-n)
df.tail(6)


# Predictors
X = df[['Close']].values[:-n]
X


X.shape


y = df['Target'].values[:-n]

# Check the output
y

y.shape


# Splitting the datasets into training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

# Output the train and test data size
print(f"Train and Test Size {len(X_train)}, {len(X_test)}")

# Scale and fit the model using pipeline
# pipe = Pipeline([("scaler", MinMaxScaler()), ("regressor", SVR(C=1, gamma='auto', kernel='linear'))]) 

pipe = Pipeline([("scaler", MinMaxScaler()), ("regressor", SVR(kernel='rbf', C=1e3, gamma=0.1))]) 
pipe.fit(X_train, y_train)

# Predicting the test dataset
y_pred = pipe.predict(X_test)
y_pred[-5:]

# Metrics 
pipe.score(X_test,y_test) # r2_score(y_test,y_pred)

# Output prediction scoare
print(f'Train Accuracy: {pipe.score(X_train,y_train):0.4}')
print(f'Test Accuracy: {pipe.score(X_test,y_test):0.4}')


# Cross-validation
tscv = TimeSeriesSplit(n_splits=5)


# Get parameters list
pipe.get_params()


# Perform Gridsearch and fit
param_grid = {"regressor__C": [0.1, 1, 10, 100, 1000],
             "regressor__kernel": ["poly", "rbf", "sigmoid"],
              "regressor__gamma": [1e-7, 1e-4, 1e-3, 1e-2]}

gs = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=tscv, verbose=1)
gs.fit(X_train, y_train)

# Best Estimator
gs.best_estimator_

# Best Params
params = gs.best_params_
params


# Best Score
# Mean cross-validated score of the best_estimator
gs.best_score_


# Predicting the test dataset
y_preds = gs.predict(X_test)
y_preds[-5:]


# Output prediction scoare
print(f'Train Accuracy\t: {gs.score(X_train,y_train):0.6}')
print(f'Test Accuracy\t: {gs.score(X_test,y_test):0.6}')


# Create a dataframe to subsume key values
df3 = pd.DataFrame({'X': X_test.flatten(), 'y': y_preds})
df3['X'] = df3['X'].shift(-n)
df3['X-y'] = df3['X'] - df3['y']
df3 = df3[:-n]

df3.tail(n)

# Check for missing values
df3.isnull().sum()


# Mean difference
print(f'Mean Difference\t: {np.mean(df3["X-y"]):0.4}')


# Plot x+5 vs y_pred
fig, ax = plt.subplots(2,2, figsize=(20,10))

ax[0,0].scatter(df3['X'], df3['y'])
ax[0,0].set_title('X+5 days vs Predicted y')

# # Plot Predicted Price
ax[0,1].plot(df3.index, y_preds[:-n], 'crimson')
ax[0,1].set_title('Predicted y')

# # Plot Residual of x+5 and y_pred 
ax[1,0].plot(df3.index, df3['X-y'])
ax[1,0].set_title('Difference in X+5 and y_preds')

# # Plot Histogram of Residual of x+5 and y_pred 
ax[1,1].hist(df3['X-y'], bins=50, density=False, color='orange')
ax[1,1].set_title('Histogram of Residual of x+5 and y_preds');

