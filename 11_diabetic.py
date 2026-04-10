#  Diabetic prediction using Decision tree regressior

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load the diabetes dataset
diabetes = load_diabetes()
# print(diabetes)
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes['target']

#  train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree Regressor model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)


#  hyperparameter tuning

params = {
     "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    "max_depth": [None, 5, 10, 20],
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'sqrt', 'log2']
}

from sklearn.model_selection import GridSearchCV

dtc = DecisionTreeRegressor()
grid = GridSearchCV(estimator=dtc , param_grid=params, cv=5, n_jobs=-1, scoring='r2')

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print("Best Parameters:", grid.best_params_)
print("Predicted values:", y_pred)

#  perfomance metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualizing the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12,8))
plot_tree(grid.best_estimator_, filled=True)
plt.title("Decision Tree Regressor Visualization")  
plt.show()
