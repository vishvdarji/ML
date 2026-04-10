from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = ...  # assume these are already defined

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LinearRegression())
])

pipe.fit(X_train, y_train)
pipe.predict(X_test)

# we use pipeline to preprocess the data and train the model in one object