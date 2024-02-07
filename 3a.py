#Step 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Step 2
df = pd.read_csv('breast_cancer.csv')
#Step 3
df = df.iloc[:, :-1]
x = df.iloc[:, 2:].values
y = df.diagnosis.values
#Step 4
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#Step 5 
from sklearn.tree import DecisionTreeClassifier
dt_classifier=DecisionTreeClassifier()
dt_classifier(x_train,y_train)
#Step 6
predictions=dt_classifier.predict(x_test)
proba_predictions=dt_classifier.predict_proba(x_test)
#Step 7
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_train,dt_classifier.predict(x_train))
accuracy_score(y_test,dt_classifier.predict(x_test))
confusion_matrix(y_train,dt_classifer.predict(x_train))
confusion_matrix(y_test,dt_classifer.predict(x_test))
classifcation_report(y_train,dt_classifier.predict(x_train))
classifcation_report(y_test,dt_classifier.predict(x_test))






