#Step 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib  inline

#Step 2
df=pd.read_csv("housing_prices.csv")

#Step 3
x=df.iloc[:,:3].values
y=df.iloc[:,3].values

#Step 4
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Step 5
from sklearn.linear_model import LinearRegression
mlr_model= LinearRegression(fit_intercept=True)
mlr_model.fit(x_train,y_train)
mlrmodel.intercept_
mlrmodel.coef_

#Step6
mlr_model.score(x_train,y_train)
mlr_model.score(x_test,y_test)
