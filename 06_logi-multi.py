import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.datasets import make_classification
import seaborn as sns

# generate a data for multi-class classification

X, y = make_classification(n_samples=1000,n_features=10,n_classes=3,n_informative=3,random_state=42)

# train and test split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size= 0.3 , random_state=42)

# model traininig

logistic = LogisticRegression()

logistic.fit(X_train , y_train)

y_pred = logistic.predict(X_test)
print("Predicted values:", y_pred)

# performance metrics
print("Accuracy score : ", accuracy_score(y_test,y_pred))
print("Confusion matrix :\n", confusion_matrix(y_test,y_pred))
print("Classification report :\n", classification_report(y_test,y_pred))

# Visualizing confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()