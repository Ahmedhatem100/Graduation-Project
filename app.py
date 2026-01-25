from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load("diabetes_survey_model.pkl")
columns = joblib.load("model_columns.pkl")

@app.get("/")
def home():
    return {"message": "Diabetes Survey API Running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]
    pred = 1 if prob > 0.4 else 0

    return {
        "diabetes": "Yes" if pred == 1 else "No",
        "probability": float(prob)
    }
