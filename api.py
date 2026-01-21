from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import bcrypt
import pickle
import numpy as np

app = FastAPI()

# ================= CORS =================
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
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()

# ================= LOAD MODEL =================
with open("credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================= SCHEMAS =================
class Auth(BaseModel):
    email: str
    password: str

class PredictInput(BaseModel):
    age: int
    income: float
    credit_score: int
    employment_type: str
    loan_amount: float
    loan_tenure: int
    past_default_history: int

# ================= AUTH =================
@app.post("/register")
def register(data: Auth):
    hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt())
    try:
        cursor.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (data.email, hashed)
        )
        conn.commit()
        return {"message": "Registered successfully"}
    except:
        raise HTTPException(status_code=400, detail="User already exists")

@app.post("/login")
def login(data: Auth):
    cursor.execute("SELECT password FROM users WHERE email=?", (data.email,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if bcrypt.checkpw(data.password.encode(), row[0]):
        return {"message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# ================= PREDICT =================
@app.post("/predict")
def predict(d: PredictInput):
    X = np.array([[
        d.age,
        d.income,
        d.credit_score,
        1 if d.employment_type == "Salaried" else 0,
        d.loan_amount,
        d.loan_tenure,
        d.past_default_history
    ]])

    prob = model.predict_proba(X)[0][1]

    if prob < 0.35:
        risk = "LOW RISK 🟢"
    elif prob < 0.60:
        risk = "MEDIUM RISK 🟡"
    else:
        risk = "HIGH RISK 🔴"

    return {
        "default_probability": float(prob),
        "risk_category": risk
    }

@app.get("/")
def health():
    return {"status": "API running"}
