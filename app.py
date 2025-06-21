# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")

app = FastAPI()

class Input(BaseModel):
    features: list  # e.g., [5.1, 3.5, 1.4, 0.2]

@app.post("/predict")
def predict(input: Input):
    data = np.array(input.features).reshape(1, -1)
    pred = model.predict(data)
    return {"prediction": int(pred[0])}
