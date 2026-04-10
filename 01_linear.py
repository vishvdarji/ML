import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_excel("weigh-heigh.xlsx")
print(df.head())

plt.scatter(df['Weight'], df['Height'])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Height vs Weight')
plt.show()

# finding correlation
print(df.corr())

# independent and dependent variables
X = df[['Weight']]   # independent variable must be in 2D
y = df['Height']     # dependent variable can be in 1D
print(X.shape)
print(y.shape)

#  train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardization
# using Z score normalization where mean=0 and std=1
scaler = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model training
regression = LinearRegression()

regression.fit(X_train,y_train)

# retrieve intercept and coefficient
print("Intercept:", regression.intercept_)
print("Coefficient:", regression.coef_)

# finding the best fit line
plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,regression.predict(X_train), color='red')
plt.show()

# prediction
y_pred = regression.predict(X_test)
print("Predicted values:", y_pred)

# comparing actual vs predicted using performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
