#Step 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Step 2
data = pd.read_csv('headbrain.csv')
#Step 3
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
#Step 4
mean_x = np.mean(X)
mean_y = np.mean(Y)
n = len(X)

numer = 0
denom = 0
for i in range(n):
   numer+=(X[i]-mean_x)*(Y[i]-mean_y)
   denom+=(Y[i]-mean_y)**2
b1=numer/denom
b2=mean_y-(b1*mean_x)
print("Coefficients")
print(b1, b0)

max_x=np.max(X)+100
min_x = np.min(X) - 100

x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

#Step 5 
ss_tot=0
ss_res=0
for i in range(n):
  y_pred=b0+b1*X[i]
  ss_res += (Y[i] - y_pred) ** 2
  ss_tot += (Y[i] - mean_y) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score")
print(r2)
  
  




  
