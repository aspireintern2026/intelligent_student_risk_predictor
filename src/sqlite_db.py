"""
sqlite_db.py
Simple SQLite persistence layer for student records and predictions.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, Any, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "data", "student_data.db")

STUDENT_COLUMNS = [
    "student_id",
    "attendance",
    "assignment_score",
    "midterm_score",
    "study_hours",
    "previous_grade",
    "quiz_scores",
    "participation",
    "sleep_hours",
    "risk_label",
    "performance_trend",
    "attendance_rate",
    "score_avg",
    "study_sleep_ratio",
]

SAMPLE_COLUMNS = [
    "student_id",
    "attendance",
    "assignment_score",
    "midterm_score",
    "study_hours",
    "previous_grade",
    "quiz_scores",
    "participation",
    "sleep_hours",
]


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            attendance REAL,
            assignment_score REAL,
            midterm_score REAL,
            study_hours REAL,
            previous_grade REAL,
            quiz_scores REAL,
            participation REAL,
            sleep_hours REAL,
            risk_label INTEGER,
            performance_trend REAL,
            attendance_rate REAL,
            score_avg REAL,
            study_sleep_ratio REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            risk_level TEXT,
            probability REAL,
            risk_score REAL,
            recommendation TEXT,
            requested_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "performance_trend" not in df.columns:
        df["performance_trend"] = (df["midterm_score"] - df["previous_grade"]).round(1)
    if "attendance_rate" not in df.columns:
        df["attendance_rate"] = (df["attendance"] / 100).round(3)
    if "score_avg" not in df.columns:
        df["score_avg"] = ((df["assignment_score"] + df["quiz_scores"]) / 2).round(1)
    if "study_sleep_ratio" not in df.columns:
        df["study_sleep_ratio"] = (df["study_hours"] / df["sleep_hours"].replace(0, 1)).round(2)
    return df


def save_students(df: pd.DataFrame) -> None:
    init_db()
    df = compute_derived_fields(df)
    rows = df[STUDENT_COLUMNS].to_dict(orient="records")

    placeholders = ", ".join(["?" for _ in STUDENT_COLUMNS])
    query = f"INSERT OR REPLACE INTO students ({', '.join(STUDENT_COLUMNS)}) VALUES ({placeholders})"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany(query, [[row[col] for col in STUDENT_COLUMNS] for row in rows])
    conn.commit()
    conn.close()


def save_prediction(student_id: str, result: Dict[str, Any]) -> None:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (student_id, risk_level, probability, risk_score, recommendation)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            student_id,
            result.get("risk_level"),
            float(result.get("probability", 0)),
            float(result.get("risk_score", 0)),
            result.get("recommendation", ""),
        ),
    )
    conn.commit()
    conn.close()


def get_stats() -> Dict[str, Any]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN risk_label = 1 THEN 1 ELSE 0 END) AS high_risk,
            SUM(CASE WHEN risk_label = 0 THEN 1 ELSE 0 END) AS low_risk,
            AVG(attendance) AS avg_attendance,
            AVG(assignment_score) AS avg_assignment
        FROM students
        """
    )
    row = cursor.fetchone()
    conn.close()

    return {
        "total_students": int(row["total"] or 0),
        "high_risk": int(row["high_risk"] or 0),
        "low_risk": int(row["low_risk"] or 0),
        "avg_attendance": round(float(row["avg_attendance"] or 0), 1),
        "avg_assignment": round(float(row["avg_assignment"] or 0), 1),
    }


def get_sample_students(n: int = 6) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT student_id, attendance, assignment_score, midterm_score,
               study_hours, previous_grade, quiz_scores, participation, sleep_hours
        FROM students
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    return [dict(row) for row in rows]


def get_recent_predictions(limit: int = 20) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT student_id, risk_level, probability, risk_score, recommendation, requested_at
        FROM predictions
        ORDER BY requested_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
