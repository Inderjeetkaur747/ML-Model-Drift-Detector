from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/latest_model.pkl")

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    try:
        arr = np.array(data.features)

        # Safety check
        if arr.shape[0] != 30:
            return {
                "error": f"Expected 30 features, got {arr.shape[0]}"
            }

        arr = arr.reshape(1, -1)

        prediction = model.predict(arr)[0]

        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}