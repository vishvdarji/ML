import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=150, n_features=5, n_informative=3, n_redundant=0, n_classes=2, random_state=42)

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
knn = KNeighborsClassifier(n_neighbors=5,algorithm='auto')   # You can change n_neighbors as needed

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Predicted values:", y_pred)

# Performance metrics
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

