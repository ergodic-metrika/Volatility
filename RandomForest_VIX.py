# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:24:43 2022

@author: sigma
"""

#source: 
#https://blog.quantinsti.com/random-forest-algorithm-in-python/    
    
#pip install quantrautil
import quantrautil as q
import numpy as np
from sklearn.ensemble import RandomForestClassifier


data = q.get_data('^VIX','2015-1-1','2022-10-26')
print(data.tail())


# Features construction 
data['Open-Close'] = (data.Open - data.Close)/data.Open
data['High-Low'] = (data.High - data.Low)/data.Low
data['percent_change'] = data['Adj Close'].pct_change()
data['std_5'] = data['percent_change'].rolling(5).std()
data['ret_5'] = data['percent_change'].rolling(5).mean()
data.dropna(inplace=True)

# X is the input variable
X = data[['Open-Close', 'High-Low', 'std_5', 'ret_5']]

# Y is the target or output variable
y = np.where(data['Adj Close'].shift(-1) > data['Adj Close'], 1, -1)

# Total dataset length
dataset_length = data.shape[0]

# Training dataset length
split = int(dataset_length * 0.65)
split


# Splitiing the X and y into train and test datasets
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Print the size of the train and test dataset
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#Training the machine learning model
clf = RandomForestClassifier(random_state=5)


# Create the model on train dataset
model = clf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
print('Correct Prediction (%): ', accuracy_score(y_test, model.predict(X_test), normalize=True)*100.0)


# Run the code to view the classification report metrics
from sklearn.metrics import classification_report
report = classification_report(y_test, model.predict(X_test))
print(report)


#Strategy returns
data['strategy_returns'] = data.percent_change.shift(-1) * model.predict(X)


%matplotlib inline
import matplotlib.pyplot as plt
data.strategy_returns[split:].hist()
plt.xlabel('Strategy returns (%)')
plt.show()


(data.strategy_returns[split:]+1).cumprod().plot()
plt.ylabel('Strategy returns (%)')
plt.show()
