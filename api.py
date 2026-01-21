from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import mysql.connector
import bcrypt
import uuid

app = FastAPI(title="Credit Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

# Database connection
def get_db():
    return mysql.connector.connect(
        host="DB_HOST",
        user="DB_USER",
        password="DB_PASSWORD",
        database="credit_risk_db"
    )

# ---------------- MODELS ----------------
class Register(BaseModel):
    email: str
    password: str

class Login(BaseModel):
    email: str
    password: str

class Applicant(BaseModel):
    income: float
    credit_score: int
    loan_amount: float
    loan_tenure: int
    employment_type: str
    past_default_history: int

# Token store (simple but valid for demo)
active_tokens = {}

def verify_token(token: str = Header(...)):
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    return active_tokens[token]

# ---------------- AUTH ----------------
@app.post("/register")
def register(user: Register):
    db = get_db()
    cur = db.cursor()

    hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt())

    try:
        cur.execute(
            "INSERT INTO banks (email, password_hash) VALUES (%s,%s)",
            (user.email, hashed.decode())
        )
        db.commit()
    except:
        raise HTTPException(status_code=400, detail="User already exists")

    return {"message": "Registered successfully"}

@app.post("/login")
def login(user: Login):
    db = get_db()
    cur = db.cursor(dictionary=True)

    cur.execute("SELECT * FROM banks WHERE email=%s", (user.email,))
    bank = cur.fetchone()

    if not bank or not bcrypt.checkpw(
        user.password.encode(),
        bank["password_hash"].encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid.uuid4())
    active_tokens[token] = bank["id"]

    return {"token": token}

# ---------------- PREDICTION ----------------
@app.post("/predict")
def predict(applicant: Applicant, bank_id: int = Depends(verify_token)):

    monthly_income = applicant.income / 12
    emi = applicant.loan_amount / applicant.loan_tenure
    emi_to_income_ratio = emi / (monthly_income + 1)
    tenure_relief = np.log(applicant.loan_tenure)

    employment_encoded = 1 if applicant.employment_type.lower() == "salaried" else 0

    df = pd.DataFrame([{
        "income": applicant.income,
        "credit_score": applicant.credit_score,
        "emi_to_income_ratio": emi_to_income_ratio,
        "tenure_relief": tenure_relief,
        "past_default_history": applicant.past_default_history,
        "employment_type_encoded": employment_encoded
    }])

    prob = float(model.predict_proba(df)[0][1])

    if prob < 0.35:
        risk = "LOW"
    elif prob < 0.60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    db = get_db()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO predictions
        (bank_id, income, credit_score, emi_to_income_ratio,
         tenure_relief, default_probability, risk_category)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """, (
        bank_id,
        applicant.income,
        applicant.credit_score,
        emi_to_income_ratio,
        tenure_relief,
        prob,
        risk
    ))
    db.commit()

    return {
        "default_probability": round(prob, 4),
        "risk_category": risk
    }

@app.get("/")
def health():
    return {"status": "API running"}
