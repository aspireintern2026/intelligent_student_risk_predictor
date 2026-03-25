"""
backend/main.py
FastAPI backend — REST API for the student risk predictor.
Run with: uvicorn backend.main:app --reload  (from project root)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import pickle

# ── Bootstrap: train model if not yet trained ─────────────────────────────────
def bootstrap():
    model_path  = os.path.join(os.path.dirname(__file__), "..", "models", "student_risk_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Models not found — running training pipeline...")
        orig = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        from src.data_pipeline import generate_synthetic_data, preprocess
        from src.train_model    import train_all
        import pandas as _pd
        os.makedirs("data/raw",       exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        df = generate_synthetic_data(500)
        df.to_csv("data/raw/student_data.csv", index=False)
        X, y, cols, _ = preprocess(df)
        proc = _pd.DataFrame(X, columns=cols)
        proc["risk_label"] = y
        proc.to_csv("data/processed/student_processed.csv", index=False)
        train_all()
        os.chdir(orig)
        print("Training complete.")

bootstrap()

# ── Change working dir so relative paths in predict.py work ──────────────────
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from src.predict import predict_risk

app = FastAPI(
    title="Student Performance Risk Predictor API",
    description="Predicts student academic risk using ML models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class StudentInput(BaseModel):
    student_id:       Optional[str]  = Field(default="S001", example="S042")
    attendance:       float = Field(..., ge=0,  le=100, example=58)
    assignment_score: float = Field(..., ge=0,  le=100, example=46)
    midterm_score:    float = Field(..., ge=0,  le=100, example=42)
    study_hours:      float = Field(..., ge=0,  le=24,  example=1.5)
    previous_grade:   float = Field(..., ge=0,  le=100, example=50)
    quiz_scores:      float = Field(..., ge=0,  le=100, example=44)
    participation:    float = Field(..., ge=0,  le=100, example=38)
    sleep_hours:      float = Field(..., ge=0,  le=24,  example=5)


class BatchInput(BaseModel):
    students: List[StudentInput]


class PredictionResult(BaseModel):
    student_id:     str
    risk_level:     str
    probability:    float
    risk_score:     float
    recommendation: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Student Risk Predictor API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict_single(student: StudentInput):
    """Predict risk for a single student."""
    try:
        data   = student.dict(exclude={"student_id"})
        result = predict_risk(data)
        return PredictionResult(student_id=student.student_id or "N/A", **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResult])
def predict_batch(batch: BatchInput):
    """Predict risk for a list of students."""
    results = []
    for s in batch.students:
        try:
            data   = s.dict(exclude={"student_id"})
            result = predict_risk(data)
            results.append(PredictionResult(student_id=s.student_id or "N/A", **result))
        except Exception as e:
            results.append(PredictionResult(
                student_id=s.student_id or "N/A",
                risk_level="Error", probability=0, risk_score=0,
                recommendation=str(e),
            ))
    return results


@app.get("/stats")
def stats():
    """Return dataset class distribution statistics."""
    try:
        df = pd.read_csv("data/raw/student_data.csv")
        return {
            "total_students": len(df),
            "high_risk":      int((df["risk_label"] == 1).sum()),
            "low_risk":       int((df["risk_label"] == 0).sum()),
            "avg_attendance": round(df["attendance"].mean(), 1),
            "avg_assignment": round(df["assignment_score"].mean(), 1),
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found. Run data_pipeline.py first.")


@app.get("/sample-students")
def sample_students():
    """Return 6 sample students for demo purposes."""
    return [
        {"student_id": "S101", "attendance": 82, "assignment_score": 78, "midterm_score": 75,
         "study_hours": 4, "previous_grade": 74, "quiz_scores": 76, "participation": 72, "sleep_hours": 7},
        {"student_id": "S102", "attendance": 58, "assignment_score": 46, "midterm_score": 42,
         "study_hours": 1.5, "previous_grade": 50, "quiz_scores": 44, "participation": 38, "sleep_hours": 5},
        {"student_id": "S103", "attendance": 70, "assignment_score": 62, "midterm_score": 60,
         "study_hours": 2.8, "previous_grade": 63, "quiz_scores": 60, "participation": 58, "sleep_hours": 6.5},
        {"student_id": "S104", "attendance": 40, "assignment_score": 32, "midterm_score": 30,
         "study_hours": 0.8, "previous_grade": 38, "quiz_scores": 33, "participation": 25, "sleep_hours": 4.5},
        {"student_id": "S105", "attendance": 92, "assignment_score": 88, "midterm_score": 85,
         "study_hours": 5.5, "previous_grade": 86, "quiz_scores": 88, "participation": 85, "sleep_hours": 7.5},
        {"student_id": "S106", "attendance": 65, "assignment_score": 55, "midterm_score": 50,
         "study_hours": 2.2, "previous_grade": 55, "quiz_scores": 53, "participation": 50, "sleep_hours": 6},
    ]
