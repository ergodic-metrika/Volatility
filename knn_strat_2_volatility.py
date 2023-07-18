# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 12:01:04 2022

@author: sigma
"""

# Base Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import statistics

# Using pyfolio
import pyfolio as pf

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, cross_val_score

# Classifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, auc, roc_curve
import yfinance as yf

asset=yf.Ticker("^VIX")
VIX = asset.history(period="4y")
VIX
# Check for missing values
VIX.isnull().sum()
df=VIX

df.describe()


#Unit root test
from unitroot import stationarity

stationarity(df['Close'])

#Half life
from HalfLife import estimate_half_life
estimate_half_life(df['Close'])

#Stationary for VIX and takes 30 days to revert to mean if we use the data from the past 15 years.

plt.plot(df['Close'], color = 'cornflowerblue', linewidth=1.0, linestyle="-")

#Features
#Features are acting as independent varaibles to determine the value of the target variable.
# Predictors: Use close-median as predictors
#df['O-C'] = df['Open'] - df['Close']
#df['H-L'] = df['High'] - df['Low']

#SD


sd=np.std(df['Close'], ddof=1)
sd


#Find median
medi=statistics.median(df['Close'])
medi

#Convert to pandas series into dataframe
VIX_data=df['Close'].to_frame()

#Moving average
df['5D SMA']=df['Close'].rolling(5).mean()
df['5D SMA']


df['20D SMA']=df['Close'].rolling(20).mean()
df['20D SMA']

df['O-C'] = df['Open'] - df['Close']
df['H-L'] = df['High'] - df['Low']

X = df[['O-C', 'H-L']].values
X[:5]

X.shape # Predictors should be of 2D

# Target

#Where pt is the current closing price of spot and pt+1 is the 1-day forward closing VIX
#If VIX>its previous closing, then long (short); else, no position or buy(sell).
y = np.where(df['Close'].shift(-1)>df['20D SMA'],-1, 0) & np.where(df['Close'].shift(-1)<df['20D SMA'],1, 0) 
y

y.shape  # Target Label should be 1D


# Splitting the datasets into training and testing data.
# Always keep shuffle = False for financial time series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

# Output the train and test data size
print(f"Train and Test Size {len(X_train)}, {len(X_test)}")

# Scale and fit the model
pipe = Pipeline([
    ("scaler", MinMaxScaler()), 
    ("classifier", KNeighborsClassifier())
]) 
pipe.fit(X_train, y_train)

# Target classes
class_names = pipe.classes_
class_names

# Target classes
class_names = pipe.classes_
class_names


# Predicting the test dataset
y_pred = pipe.predict(X_test)

acc_train = accuracy_score(y_train, pipe.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)

print(f'Train Accuracy: {acc_train:0.4}, Test Accuracy: {acc_test:0.4}')


# Confusion Matrix for binary classification
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)

# Plot confusion matrix
plot_confusion_matrix(pipe, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.grid(False)

# Predict Probabilities
probs = pipe.predict_proba(X_test)
preds1 = probs[:, 0]
preds2 = probs[:, 1]

fpr1, tpr1, threshold1 = roc_curve(y_test, preds1, pos_label=-1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, threshold2 = roc_curve(y_test, preds2, pos_label=1)
roc_auc2 = auc(fpr2, tpr2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

ax[0].plot([0, 1], [0, 1], 'r--')
ax[0].plot(fpr1, tpr1, 'cornflowerblue', label=f'AUC = {roc_auc1:0.2}')
ax[0].set_title("Receiver Operating Characteristic for Down Moves")
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')

ax[1].plot([0, 1], [0, 1], 'r--')
ax[1].plot(fpr2, tpr2, 'cornflowerblue', label=f'AUC = {roc_auc2:0.2}')
ax[1].set_title("Receiver Operating Characteristic for Up Moves")
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')

# Define legend
ax[0].legend(), ax[1].legend();

# Classification Report
print(classification_report(y_test, y_pred))

# First 3 split
tscv = TimeSeriesSplit(n_splits=3)
for train, test in tscv.split(X):
    print(train, test)
    
# Get parameters list
pipe.get_params()

# Perform Gridsearch and fit
param_grid = {"classifier__n_neighbors": np.arange(1,51,1)}

grid_search = GridSearchCV(pipe, param_grid, scoring='roc_auc', n_jobs=-1, cv=tscv, verbose=1)
grid_search.fit(X_train, y_train)

# Best Params
grid_search.best_params_

# Best Score
grid_search.best_score_


error_rate = []
acc_score = []

for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    acc_score.append(accuracy_score(y_test, pred_i))
    
fig, ax1 = plt.subplots(figsize=(20,10))

ax2 = ax1.twinx()
ax1.plot(range(1,51), error_rate, color= 'crimson', linestyle='dashed',  marker='o', markerfacecolor='black', markersize=5)
ax2.plot(range(1,51), acc_score, color='cornflowerblue', linestyle='dashed', marker='o', markerfacecolor='black', markersize=5)

ax1.set_xlabel('K')
ax1.set_ylabel('Error Rate', color='crimson')
ax2.set_ylabel('Accuracy Score', color='cornflowerblue')

fig.suptitle('Error Rate & Accuracy Score vs. K Value');

# Instantiate KNN model with search param
clf = KNeighborsClassifier(n_neighbors = grid_search.best_params_['classifier__n_neighbors'])
# Fit the model
clf.fit(X_train, y_train)  

# Predicting the test dataset
y_pred = clf.predict(X_test)


# Measure Accuracy
acc_train = accuracy_score(y_train, clf.predict(X_train))
acc_test = accuracy_score(y_test, y_pred)
# Print Accuracy
print(f'\n Training Accuracy \t: {acc_train :0.4} \n Test Accuracy \t\t: {acc_test :0.4}')
# Confusion Matrix for binary classification
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)

# Plot confusion matrix
plot_confusion_matrix(clf, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.grid(False)


# Predict Probabilities
probs = clf.predict_proba(X_test)
preds1 = probs[:, 0]
preds2 = probs[:, 1]

fpr1, tpr1, threshold1 = roc_curve(y_test, preds1, pos_label=-1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, threshold2 = roc_curve(y_test, preds2, pos_label=1)
roc_auc2 = auc(fpr2, tpr2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

ax[0].plot([0, 1], [0, 1], color= 'crimson', linestyle='dashed')
ax[0].plot(fpr1, tpr1, 'cornflowerblue', label=f'AUC = {roc_auc1:0.2}')
ax[0].set_title("Receiver Operating Characteristic for Down Moves")
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')

ax[1].plot([0, 1], [0, 1], color= 'crimson', linestyle='dashed')
ax[1].plot(fpr2, tpr2, 'cornflowerblue', label=f'AUC = {roc_auc2:0.2}')
ax[1].set_title("Receiver Operating Characteristic for Up Moves")
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')

# Define legend
ax[0].legend(), ax[1].legend();

# Classification Report
print(classification_report(y_test, y_pred))


#Trading Strategy
df1 = df[-len(X_test):]
df1['Signal'] = clf.predict(X_test)


# Buy & Hold
df1['Returns'] = np.log(df1['Close']).diff().fillna(0)
cumret = df1['Returns'].cumsum().apply(np.exp)

# KNN Algorithm
df1['Strategy'] = df1['Returns'] * df1['Signal'].shift(1).fillna(0)
cumstg = df1['Strategy'].cumsum().apply(np.exp)

# Check the output
df1.tail(20)


# Plot graph iteratively
fig, ax = plt.subplots(2,2, figsize=(20,10))

# 2019
ax[0,0].plot(cumret['2019'], label ='VIX', color ='cornflowerblue')
ax[0,0].plot(cumstg['2019'], label ='Strategy', color ='crimson')
# 2020
ax[0,1].plot(cumret['2020'], label ='VIX', color ='cornflowerblue')
ax[0,1].plot(cumstg['2020'], label ='Strategy', color ='crimson')
# 2021
ax[1,0].plot(cumret['2021'], label ='VIX', color ='cornflowerblue')
ax[1,0].plot(cumstg['2021'], label ='Strategy', color ='crimson')
# 2022
ax[1,1].plot(cumret['2022'], label ='VIX', color ='cornflowerblue')
ax[1,1].plot(cumstg['2022'], label ='Strategy', color ='crimson')
ax[1,1].tick_params(axis='x', rotation=90)


# Set axis title
ax[0,0].set_title('2019'), ax[0,1].set_title('2020'), ax[1,0].set_title('2021'), ax[1,1].set_title('2022')

# Define legend
ax[0,0].legend(), ax[0,1].legend(), ax[1,0].legend(), ax[1,1].legend()

fig.suptitle('Trading Strategy Performance', fontsize=14);

# Cumulative returns
print(' Benchmark & Strategy Returns')
print(f'\n Benchmark Return \t: {cumret[-1]:0.4} \n KNN Strategy Return \t: {cumstg[-1]:0.4}')



# Create Tear sheet using pyfolio for outsample
pf.create_simple_tear_sheet(df1['Strategy'])
