"""
train_model.py
Trains and evaluates multiple models, saves the best one.
"""

import pandas as pd
import numpy as np
import pickle, os
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, accuracy_score)
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost model.")

try:
    from src.sqlite_db import DB_PATH, init_db
except ImportError:
    import sqlite_db as _sqlite
    DB_PATH = _sqlite.DB_PATH
    init_db = _sqlite.init_db


RANDOM_STATE = 42


def load_data():
    init_db()
    query = (
        "SELECT attendance, assignment_score, midterm_score, study_hours, previous_grade, "
        "quiz_scores, participation, sleep_hours, performance_trend, attendance_rate, score_avg, "
        "study_sleep_ratio, risk_label FROM students"
    )
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            raise ValueError("SQLite database has no student records")
    except Exception:
        df = pd.read_csv("data/processed/student_processed.csv")

    X = df.drop("risk_label", axis=1).values
    y = df["risk_label"].values
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, preds)
    f1     = f1_score(y_test, preds)
    auc    = roc_auc_score(y_test, probas)
    cv     = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  CV F1    : {cv:.4f}")
    print(classification_report(y_test, preds, target_names=["Low Risk", "High Risk"]))
    return {"name": name, "model": model, "accuracy": acc, "f1": f1, "auc": auc}


def train_all():
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    models_to_train = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE)),
        ("Gradient Boosting",
         GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=RANDOM_STATE)),
    ]
    if XGBOOST_AVAILABLE:
        models_to_train.append((
            "XGBoost",
            XGBClassifier(n_estimators=200, learning_rate=0.1,
                          use_label_encoder=False, eval_metric="logloss",
                          random_state=RANDOM_STATE)
        ))

    results = []
    for name, m in models_to_train:
        r = evaluate_model(name, m, X_train, X_test, y_train, y_test)
        results.append(r)

    best = max(results, key=lambda x: x["f1"])
    print(f"\n Best model: {best['name']}  (F1={best['f1']:.4f}, AUC={best['auc']:.4f})")

    os.makedirs("models", exist_ok=True)
    with open("models/student_risk_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    print("Model saved → models/student_risk_model.pkl")

    # Save comparison CSV
    os.makedirs("reports", exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results]
                 ).to_csv("reports/model_comparison.csv", index=False)
    print("Comparison saved → reports/model_comparison.csv")
    return best


if __name__ == "__main__":
    train_all()
