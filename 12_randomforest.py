import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error

df = pd.read_csv("cardekho.csv")
print(df.head())

print(df.info())
print(df.describe())

print(df.isnull().sum())

print(df["owner"].value_counts())
print(df["transmission"].value_counts())
print(df["seller_type"].value_counts())
print(df["fuel"].value_counts())

# train and test split
X = df.drop(["selling_price"],axis=1)
y = df["selling_price"]

#  label encoding
le = LabelEncoder()
X["name"] = le.fit_transform(X["name"])

print(X.head())


#  we use one hot encoding for categorical variables with few unique values
#   Column transformer with numeric and categorical features

onehot_cols = ["fuel","seller_type","transmission","owner"]
num_feature = X.select_dtypes(exclude= "object").columns

numeric_features = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
        ('StandardScaler', numeric_features, num_feature),
        ('OneHotEncoder', oh_transformer, onehot_cols)
    ], remainder='passthrough'
)

X = preprocessor.fit_transform(X)
print(X.shape)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(real , predicted):
    mse = mean_squared_error(real, predicted)
    mae = mean_absolute_error(real, predicted)
    r2 = r2_score(real, predicted)
    return mse, mae, r2

# beginning model training
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "KNN Regressor": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor()
}

# with the help of loop train multiple model with train data and find the perfomance metrices
# whichever model gives best score we used model and do hyperparameter tuning