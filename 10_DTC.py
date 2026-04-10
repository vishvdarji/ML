#  Decision tree classifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Load the iris dataset
iris = load_iris()
# print(iris['DESCR'])

#  independent features
X = pd.DataFrame(iris['data'], columns=iris.feature_names)
# print(X.head())

#  dependent feature
y = iris['target']

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
dtc = DecisionTreeClassifier(max_depth=3)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("Predicted values:", y_pred)

# Performance metrics
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualizing the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12,8))
plot_tree(dtc, filled=True)
plt.title("Decision Tree Visualization")
# plt.show()



#  prepuning and hyperparameter tuning

params = {
    'max_depth': [2, 3, 4, 5, 6],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}

from sklearn.model_selection import GridSearchCV

dtc = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dtc, param_grid=params, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)
y_pred_best = grid_search.predict(X_test)      
print("Best Parameters:", grid_search.best_params_)

