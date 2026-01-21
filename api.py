from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import bcrypt
import pickle
import numpy as np
import uuid

# ================= APP =================
app = FastAPI(title="Credit Risk API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://credit-risk-detect.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= DATABASE =================
DB = "users.db"

def get_db():
    return sqlite3.connect(DB, check_same_thread=False)

conn = get_db()
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    password TEXT
)
""")

conn.commit()

# ================= TOKEN STORE =================
active_tokens = {}

# ================= LOAD MODEL =================
with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================= SCHEMAS =================
class Auth(BaseModel):
    email: str
    password: str

class PredictInput(BaseModel):
    income: float
    credit_score: int
    loan_amount: float
    loan_tenure: int
    employment_type: str
    past_default_history: int

# ================= REGISTER =================
@app.post("/register")
def register(data: Auth):
    hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()

    try:
        cur.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (data.email, hashed)
        )
        conn.commit()
        return {"message": "Registered successfully"}
    except:
        raise HTTPException(status_code=400, detail="User already exists")

# ================= LOGIN =================
@app.post("/login")
def login(data: Auth):
    cur.execute("SELECT id, password FROM users WHERE email=?", (data.email,))
    row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id, stored_hash = row

    if not bcrypt.checkpw(data.password.encode(), stored_hash.encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid.uuid4())
    active_tokens[token] = user_id

    return {"token": token}

# ================= TOKEN CHECK =================
def verify_token(token: str | None):
    if not token or token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return active_tokens[token]

# ================= PREDICT =================
import pandas as pd

@app.post("/predict")
def predict(data: PredictInput, token: str = Header(None)):
    verify_token(token)

    # Recreate the EXACT training schema
    df = pd.DataFrame([{
        "income": data.income,
        "credit_score": data.credit_score,
        "loan_amount": data.loan_amount,
        "loan_tenure": data.loan_tenure,
        "employment_type": data.employment_type,
        "past_default_history": data.past_default_history
    }])

    # Let the pipeline handle preprocessing
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


