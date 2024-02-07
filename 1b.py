#Step 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Step 2
df=pd.read_csv("housing_prices_SLR.csv",delimiter=',')
plt.scatter(df.AREA,df.PRICE,c='blue')
plt.show()
plt.scatter(df.AREA,df.PRICE,c=np.random.random(df.shape[0]))
plt.show()
col=np.random.random(df.shape[0])
plt.scatter(df.AREA,df.PRICE,c=col,s=4)
plt.show()
#Step 3
x=df[['AREA']].values#feature Matrix
y=df.PRICE.values#Target Matrix
#Step 4
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100) #80 20 split,random_state to reproduce the same split everytime
print(x_train.shape)
print(x_test.shape)
print(x_train.shape)
print(x_test.shape)
#Step 5
from sklearn.linear_model import LinearRegression
lr_model= LinearRegression()
lr_model.fit(x_train,y_train)
print(lr_model.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(lr_model.coef_)
lr_model=LinearRegression(fit_intercept= False)
lr_model.fit(x_train,y_train)
print(lr_model.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(lr_model.coef_)#y=c+mx
#Step 6
lr_model.predict(x_train)
#Step 7
from sklearn.metrics import r2_score
r2_score(y_train,lr_model.predict(x_train)
r2_score(y_test,lr_model.predict(x_test)
lr_model.score(x_test,y_test)

#Step 8
plt.scatter(x_train,y_train,c='red')
plt.scatter(x_test,y_test,c='blue')
plt.plot(x_test,lr_model.predict(x_test),c='y')
  
