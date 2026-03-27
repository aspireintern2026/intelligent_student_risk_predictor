"""
data_pipeline.py
Generates synthetic student data and preprocesses it for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

try:
    from src.sqlite_db import init_db, save_students
except ImportError:
    import sqlite_db as _sqlite
    init_db = _sqlite.init_db
    save_students = _sqlite.save_students

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_synthetic_data(n_students: int = 500) -> pd.DataFrame:
    """Generate realistic synthetic student academic data."""

    # High-risk students (~35% of dataset)
    n_high = int(n_students * 0.35)
    n_low = n_students - n_high

    def make_students(n, risk):
        if risk == 1:  # High risk
            attendance       = np.random.normal(55,  15, n).clip(10, 75)
            assignment_score = np.random.normal(48,  12, n).clip(10, 65)
            midterm_score    = np.random.normal(44,  14, n).clip(10, 65)
            study_hours      = np.random.normal(1.5,  0.8, n).clip(0,  4)
            prev_grade       = np.random.normal(52,  12, n).clip(20, 65)
            quiz_scores      = np.random.normal(45,  13, n).clip(10, 65)
            participation    = np.random.normal(40,  15, n).clip(5,  65)
            sleep_hours      = np.random.normal(5.5,  1.2, n).clip(3,  7)
        else:           # Low risk
            attendance       = np.random.normal(82,  10, n).clip(60, 100)
            assignment_score = np.random.normal(75,  10, n).clip(55, 100)
            midterm_score    = np.random.normal(72,  12, n).clip(50, 100)
            study_hours      = np.random.normal(4.0,  1.2, n).clip(2,   9)
            prev_grade       = np.random.normal(74,  10, n).clip(55, 100)
            quiz_scores      = np.random.normal(72,  10, n).clip(50, 100)
            participation    = np.random.normal(72,  12, n).clip(50, 100)
            sleep_hours      = np.random.normal(7.0,  1.0, n).clip(5,   9)

        ids = [f"S{i+1:03d}" for i in range(n)]
        return pd.DataFrame({
            "student_id":       ids,
            "attendance":       np.round(attendance, 1),
            "assignment_score": np.round(assignment_score, 1),
            "midterm_score":    np.round(midterm_score, 1),
            "study_hours":      np.round(study_hours, 1),
            "previous_grade":   np.round(prev_grade, 1),
            "quiz_scores":      np.round(quiz_scores, 1),
            "participation":    np.round(participation, 1),
            "sleep_hours":      np.round(sleep_hours, 1),
            "risk_label":       risk,
        })

    df_high = make_students(n_high, 1)
    df_low  = make_students(n_low,  0)
    df_high["student_id"] = [f"S{i+1:03d}"    for i in range(n_high)]
    df_low["student_id"]  = [f"S{n_high+i+1:03d}" for i in range(n_low)]

    df = pd.concat([df_high, df_low], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def preprocess(df: pd.DataFrame):
    """Clean, engineer features, and scale the dataset."""
    df = df.dropna().copy()

    # Derived features
    df["performance_trend"]    = (df["midterm_score"] - df["previous_grade"]).round(1)
    df["attendance_rate"]      = (df["attendance"] / 100).round(3)
    df["score_avg"]            = ((df["assignment_score"] + df["quiz_scores"]) / 2).round(1)
    df["study_sleep_ratio"]    = (df["study_hours"] / df["sleep_hours"].replace(0, 1)).round(2)

    feature_cols = [
        "attendance", "assignment_score", "midterm_score",
        "study_hours", "previous_grade", "quiz_scores",
        "participation", "sleep_hours",
        "performance_trend", "attendance_rate", "score_avg", "study_sleep_ratio",
    ]

    X = df[feature_cols].values
    y = df["risk_label"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)

    try:
        save_students(df)
    except Exception:
        pass

    return X_scaled, y, feature_cols, scaler


if __name__ == "__main__":
    os.makedirs("data/raw",       exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Generating synthetic student data...")
    df = generate_synthetic_data(500)
    df.to_csv("data/raw/student_data.csv", index=False)
    print(f"Raw data saved → data/raw/student_data.csv  ({len(df)} rows)")

    X, y, cols, scaler = preprocess(df)
    processed = pd.DataFrame(X, columns=cols)
    processed["risk_label"] = y
    processed.to_csv("data/processed/student_processed.csv", index=False)
    print(f"Processed data saved → data/processed/student_processed.csv")
    print(f"Features: {cols}")
    print(f"Class distribution — Low risk: {(y==0).sum()}  High risk: {(y==1).sum()}")
