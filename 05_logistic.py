import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import RandomizedSearchCV

#  create a dataset for binary classification
X,y = make_classification(n_samples=1000, n_features=10, n_classes=2,random_state=42)

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
logistic = LogisticRegression()

logistic.fit(X_train,y_train)

y_pred=logistic.predict(X_test)
print("Predicted values:", y_pred)


# performance metrics
print("Accuracy Score:", accuracy_score(y_test,y_pred))

print("confuasion matrix :\n", confusion_matrix(y_test,y_pred))

print("Classifiaction report l:\n", classification_report(y_test,y_pred))



#  HYPERPARAMETER TUNING & CROSS-VALIDATION
 
#  --> used to improve the model perfomance by finding the best set of parameters for the model

penalty = ['l1', 'l2', 'elasticnet', 'none']
C = [0.01, 0.1, 1, 10, 100]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

params = {'penalty': penalty, 'C': C, 'solver': solver}
print("Parameters for tuning:", params)

# cross-validation strategy
cv = StratifiedGroupKFold()

# hyperparameter tuning using GridSearchCV
grid = GridSearchCV(estimator=logistic, param_grid=params, scoring='accuracy', cv=cv,n_jobs=-1)

grid.fit(X_train , y_train)

print("Best parameters found:", grid.best_params_)

print("Best accuracy found:", grid.best_score_)


# hyperparameter tuning usig RandomizedSearchCV

randomCV = RandomizedSearchCV(estimator=logistic, param_distributions=params, scoring='accuracy', cv=cv,n_jobs=-1)

randomCV.fit(X_train,y_train)

print("Best parameters found (RandomizedSearchCV):", randomCV.best_params_)

print("Best accuracy found (RandomizedSearchCV):", randomCV.best_score_)


# DIFFERENCE BETWEEN GridSearchCV AND RandomizedSearchCV

#  GridSearchCV
# Tries ALL combinations
# Accurate but slow

# RandomizedSearchCV
# Tries RANDOM combinations
# Faster, good for large search space