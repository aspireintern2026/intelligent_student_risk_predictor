"""
predict.py
Prediction engine — loads the trained model and returns risk predictions.
"""

import pickle
import numpy as np
from typing import Dict, Any


def load_artifacts():
    with open("models/student_risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)
    return model, scaler_data["scaler"], scaler_data["feature_cols"]


def build_features(data: Dict[str, float], feature_cols):
    """Compute all features including derived ones from raw student input."""
    attendance       = float(data.get("attendance", 75))
    assignment_score = float(data.get("assignment_score", 70))
    midterm_score    = float(data.get("midterm_score", 65))
    study_hours      = float(data.get("study_hours", 3))
    previous_grade   = float(data.get("previous_grade", 65))
    quiz_scores      = float(data.get("quiz_scores", 70))
    participation    = float(data.get("participation", 65))
    sleep_hours      = float(data.get("sleep_hours", 7))

    # Derived
    performance_trend = midterm_score - previous_grade
    attendance_rate   = attendance / 100
    score_avg         = (assignment_score + quiz_scores) / 2
    study_sleep_ratio = study_hours / max(sleep_hours, 0.1)

    raw = {
        "attendance":        attendance,
        "assignment_score":  assignment_score,
        "midterm_score":     midterm_score,
        "study_hours":       study_hours,
        "previous_grade":    previous_grade,
        "quiz_scores":       quiz_scores,
        "participation":     participation,
        "sleep_hours":       sleep_hours,
        "performance_trend": performance_trend,
        "attendance_rate":   attendance_rate,
        "score_avg":         score_avg,
        "study_sleep_ratio": study_sleep_ratio,
    }
    return np.array([[raw[c] for c in feature_cols]])


def get_recommendation(prob: float, data: Dict) -> str:
    attendance = data.get("attendance", 75)
    assignment  = data.get("assignment_score", 70)
    study_hrs   = data.get("study_hours", 3)

    tips = []
    if attendance < 65:
        tips.append("Improve attendance (currently below 65%)")
    if assignment < 55:
        tips.append("Complete all pending assignments")
    if study_hrs < 2:
        tips.append("Increase daily study hours to at least 3 hrs")
    if prob >= 0.75:
        tips.append("Schedule immediate academic counseling session")
    elif prob >= 0.5:
        tips.append("Enroll in peer tutoring or study group")

    if not tips:
        tips = ["Maintain current performance", "Attend optional revision sessions"]

    return " | ".join(tips)


def predict_risk(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function.

    Parameters
    ----------
    student_data : dict with keys:
        attendance, assignment_score, midterm_score, study_hours,
        previous_grade, quiz_scores, participation, sleep_hours

    Returns
    -------
    dict: risk_level, probability, risk_score, recommendation
    """
    model, scaler, feature_cols = load_artifacts()
    X_raw    = build_features(student_data, feature_cols)
    X_scaled = scaler.transform(X_raw)
    pred     = int(model.predict(X_scaled)[0])
    prob     = float(model.predict_proba(X_scaled)[0][1])
    risk_score = round(prob * 100, 1)

    if prob >= 0.70:
        risk_level = "High"
    elif prob >= 0.45:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "risk_level":      risk_level,
        "probability":     round(prob, 4),
        "risk_score":      risk_score,
        "recommendation":  get_recommendation(prob, student_data),
    }


if __name__ == "__main__":
    test = {
        "attendance": 58, "assignment_score": 46, "midterm_score": 42,
        "study_hours": 1.5, "previous_grade": 50, "quiz_scores": 44,
        "participation": 38, "sleep_hours": 5,
    }
    result = predict_risk(test)
    print("=== Prediction Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
