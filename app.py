#!/usr/bin/env python3
"""
setup_and_run.py
One-click setup: generates data, trains the model, then starts the API.
Run: python setup_and_run.py
"""
import os, sys, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

def step(msg):
    print(f"\n{'─'*50}\n  {msg}\n{'─'*50}")

step("Step 1 — Generating synthetic student data")
from src.data_pipeline import generate_synthetic_data, preprocess
import pandas as pd

os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models",         exist_ok=True)
os.makedirs("reports",        exist_ok=True)

df = generate_synthetic_data(500)
df.to_csv("data/raw/student_data.csv", index=False)
print(f"  Generated {len(df)} students")

X, y, cols, scaler = preprocess(df)
proc = pd.DataFrame(X, columns=cols)
proc["risk_label"] = y
proc.to_csv("data/processed/student_processed.csv", index=False)
print(f"  Saved processed data ({len(cols)} features)")

step("Step 2 — Training ML models")
from src.train_model import train_all
best = train_all()
print(f"  Best model: {best['name']}  F1={best['f1']:.4f}")

step("Step 3 — Starting FastAPI backend")
print("  API will be available at: http://127.0.0.1:8000")
print("  API docs:                 http://127.0.0.1:8000/docs")
print("  Open frontend/index.html in your browser")
print("\n  Press Ctrl+C to stop the server\n")

subprocess.run([
    sys.executable, "-m", "uvicorn",
    "backend.main:app",
    "--reload",
    "--host", "127.0.0.1",
    "--port", "8000",
])
