"""FastAPI production endpoint for churn prediction."""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import os

app = FastAPI(
    title="Telco Churn Early Warning API",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/phase4_logistic_regression.pkl")
pipeline = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_tier: Literal["Critical", "High", "Medium", "Low"]
    will_churn: bool


def tier(proba: float) -> str:
    if proba >= 0.7: return "Critical"
    elif proba >= 0.4: return "High"
    elif proba >= 0.2: return "Medium"
    else: return "Low"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer):
    try:
        df = pd.DataFrame([customer.model_dump()])
        proba = float(pipeline.predict_proba(df)[0, 1])
        return PredictionResponse(
            churn_probability=round(proba, 4),
            risk_tier=tier(proba),
            will_churn=proba >= 0.5
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    