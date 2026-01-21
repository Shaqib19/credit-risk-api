from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

# ==================================================
# LOAD MODEL
# ==================================================
with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Credit Risk API")

# ==================================================
# CORS (REQUIRED FOR NETLIFY)
# ==================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to Netlify URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# INPUT SCHEMA
# ==================================================
class Applicant(BaseModel):
    age: int
    income: float
    credit_score: int
    employment_type: str
    loan_amount: float
    loan_tenure: int
    past_default_history: int

# ==================================================
# PREDICTION
# ==================================================
@app.post("/predict")
def predict(applicant: Applicant):

    df = pd.DataFrame([applicant.dict()])

    # Feature engineering (MATCH TRAINING)
    df["monthly_income"] = df["income"] / 12
    df["emi"] = df["loan_amount"] / df["loan_tenure"]
    df["emi_to_income_ratio"] = df["emi"] / (df["monthly_income"] + 1)
    df["tenure_relief"] = np.log(df["loan_tenure"])

    df = df.drop(columns=["loan_amount"])

    prob = model.predict_proba(df)[0][1]

    if prob < 0.35:
        risk = "LOW"
    elif prob < 0.60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "default_probability": round(float(prob), 4),
        "risk_category": risk
    }

@app.get("/")
def health():
    return {"status": "API running"}
