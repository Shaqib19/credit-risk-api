from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

# Load model
with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Credit Risk API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Applicant(BaseModel):
    age: int
    income: float
    credit_score: int
    employment_type: str
    loan_amount: float
    loan_tenure: int
    past_default_history: int

@app.post("/predict")
def predict(applicant: Applicant):

    # HARD SAFETY CHECK
    if (
        applicant.income <= 0 or
        applicant.loan_amount <= 0 or
        applicant.loan_tenure <= 0 or
        applicant.credit_score <= 0
    ):
        return {"default_probability": 1.0, "risk_category": "HIGH"}

    df = pd.DataFrame([applicant.dict()])

    df["monthly_income"] = df["income"] / 12
    df["emi"] = df["loan_amount"] / df["loan_tenure"]
    df["emi_to_income_ratio"] = df["emi"] / (df["monthly_income"] + 1)
    df["tenure_relief"] = np.log(df["loan_tenure"])

    df = df.drop(columns=["loan_amount"])

    # FINAL NAN CHECK
    if df.isna().any().any() or np.isinf(df.values).any():
        return {"default_probability": 1.0, "risk_category": "HIGH"}

    prob = float(model.predict_proba(df)[0][1])

    if prob < 0.35:
        risk = "LOW"
    elif prob < 0.60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "default_probability": round(prob, 4),
        "risk_category": risk
    }

@app.get("/")
def health():
    return {"status": "API running"}
