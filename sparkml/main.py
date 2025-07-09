from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):
    X = np.array([list(data.values())])
    prediction = model.predict(X)
    return {"result": prediction.tolist()}
