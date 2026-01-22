from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3, bcrypt, pickle, uuid
import pandas as pd

# ================= APP =================
app = FastAPI(title="Credit Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    age: int = 35
    income: float
    credit_score: int
    employment_type: str
    loan_amount: float
    loan_tenure: int
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

# ================= PREDICT =================
@app.post("/predict")
def predict(data: PredictInput, authorization: str = Header(None)):
    verify_token(authorization)

    try:
        # ================= FEATURE ENGINEERING (MATCH TRAINING) =================
        monthly_income = data.income / 12
        emi = data.loan_amount / max(data.loan_tenure, 1)

        loan_to_income_ratio = data.loan_amount / max(data.income, 1)
        tenure_relief = 1 if data.loan_tenure >= 24 else 0

        # ================= EXACT MODEL INPUT =================
        df = pd.DataFrame([{
            "monthly_income": monthly_income,
            "emi": emi,
            "credit_score": data.credit_score,
            "employment_type": data.employment_type,
            "loan_amount": data.loan_amount,
            "loan_tenure": data.loan_tenure,
            "loan_to_income_ratio": loan_to_income_ratio,
            "tenure_relief": tenure_relief,
            "past_default_history": data.past_default_history
        }])

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

    except Exception as e:
        print("🔥 PREDICT ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
