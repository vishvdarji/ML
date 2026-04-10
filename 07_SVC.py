import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

X,y = make_classification(n_samples=1000,n_features=2,n_classes=2,n_clusters_per_class=1,n_redundant=0,random_state=42)

#  train and test split
X_train , X_test , y_train ,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

# model training
svc = SVC()

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)
print(f"predicted output is : {y_pred}")

#  perfomance matrics

print(f"classifiction report is : \n {classification_report(y_test,y_pred)}")
print(f"confusion metrics is : \n {confusion_matrix(y_test,y_pred)}")


#  we get different decision boundaries by changing the kernel parameter

# we can chhose best kernal by using hyperparameter tuning and cross-validation

#  same as in logistic regression example