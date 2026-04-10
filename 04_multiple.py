import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score


df = pd.read_excel("house_price.xlsx")
print(df.head())
df.drop("No",axis=1,inplace=True)
print(df.head())

#  visualizations

# sns.pairplot(df)
# plt.show()

sns.regplot(x='Size', y='Price', data=df)       # regplot to show linear relationship
plt.title('Size vs Price')
sns.regplot(x='Bedrooms', y='Price', data=df)
plt.show()

# Correlation matrix
print(df.corr())


# independent and dependent variables and train test split
X=df[["Size","Bedrooms"]]
y=df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# model training
regression = LinearRegression()
regression.fit(X_train,y_train)

# retrieve intercept and coefficient
print("Intercept:", regression.intercept_)
print("Coefficients:", regression.coef_)

print("Predicted values:", regression.predict(X_test))


# perdformance metrics
print("Mean Squared Error:", mean_squared_error(y_test, regression.predict(X_test)))
print("Mean Absolute Error:", mean_absolute_error(y_test, regression.predict(X_test)))
print("R2 Score:", r2_score(y_test, regression.predict(X_test)))

