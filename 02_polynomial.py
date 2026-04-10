import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# generating polynomial data
X = 6 * np.random.rand(100,1) - 3
y = 0.5 * X**2 + 1.5*X + 2 + np.random.randn(100,1)   #quadratic equation with outliers 


plt.scatter(X,y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Data')
plt.show()

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# first we try without polynomial features
regression_1 = LinearRegression()
regression_1.fit(X_train,y_train) 
y_predi=regression_1.predict(X_test)
sscore = print(r2_score(y_test,y_predi))
print("R2 Score without Polynomial Features:", sscore)  

#transforming to polynomial features  

poly =  PolynomialFeatures(degree=2,include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# model training
regression = LinearRegression()
regression.fit(X_train_poly,y_train)

y_pred=regression.predict(X_test_poly)

score = r2_score(y_test,y_pred)
print("R2 Score:", score)