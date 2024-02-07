#Step 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
#Step 2
df=pd.read_csv("breast_cancer.csv")
#Step 3
df=df.iloc[:,:-1]
x=df.iloc[:,2:].values
y=df.diagnosis.values
#Step 4
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=500)
#Step 5
(y_train == 'M').sum()
(y_train=='B').sum()
278/len(y_train)  # Baseline model of accuracy =(more number of occurrences)/total data elements
#Step 6
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
baseline_pred=["B"]*len(y_train)
accuracy_score(y_train,baseline_pred)
confusion_matrix(y_train,baseline_pred)
classification_report(y_train,baseline_pred)
#Step 7
from sklearn.naive_bayes import GaussianNB
nb_model=GaussianNB()
nb_model.fit(x_train,y_train)
#Step 8
nb_model.score(x_train,y_train)
nb_model.score(x_test,y_test)
confusion_matrix(y_train,nb_model.predict(x_train))
confusion_matrix(y_test,nb_model.predict(x_test))
print(classification_report(y_train,nb_model.predict(x_train)))
print(classification_report(y_test,nb_model.predict(x_test)))
