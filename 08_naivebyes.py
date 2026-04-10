from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

X,y = load_iris(return_X_y = True)

# train and test split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# model training
gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
print("Predicted values:", y_pred)

# performance metrics

print("Accuracy :", accuracy_score(y_test,y_pred))

print("Classification Report:\n", classification_report(y_test,y_pred))    