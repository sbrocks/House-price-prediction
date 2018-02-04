# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#Importing the dataset
dataset1=pd.read_csv('train.csv')
X_train=dataset1.iloc[:,1:14].values
y_train=dataset1.iloc[:,14].values

dataset2=pd.read_csv('test.csv')
X_test=dataset2.iloc[:,1:14].values

"""
# Splitting tha dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
"""

# Fitting the Random forest regressor 
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=50,random_state=0)
regressor.fit(X_train,y_train)

# Predicting the house price
y_pred=regressor.predict(X_test)
print(y_pred)

