from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

X, y = load_iris(return_X_y=True)
model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
