# Student Performance Risk Predictor

AI-powered system that predicts which students are at academic risk — enabling early intervention.

## Project Structure

```
student_risk_predictor/
├── backend/
│   └── main.py                  ← FastAPI REST API
├── frontend/
│   └── index.html               ← Teacher Dashboard (open in browser)
├── src/
│   ├── data_pipeline.py         ← Data generation + preprocessing
│   ├── train_model.py           ← Model training + evaluation
│   └── predict.py               ← Prediction engine
├── data/
│   ├── raw/                     ← Raw synthetic student data
│   └── processed/               ← Scaled feature data
├── models/                      ← Saved ML model + scaler
├── reports/                     ← Model comparison CSV
├── requirements.txt
└── setup_and_run.py             ← One-click setup script
```

## Quick Start (VS Code)

### Step 1 — Open project in VS Code
- Open VS Code
- File → Open Folder → select `student_risk_predictor/`

### Step 2 — Open Terminal
- Press `Ctrl + `` ` (backtick) to open integrated terminal

### Step 3 — Create virtual environment
```bash
python -m venv venv
```

### Step 4 — Activate virtual environment
**Windows:**
```bash
venv\Scripts\activate
```
**Mac / Linux:**
```bash
source venv/bin/activate
```

### Step 5 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 6 — Run everything (data + training + API)
```bash
python setup_and_run.py
```

This will:
1. Generate 500 synthetic student records
2. Train Logistic Regression, Random Forest, and XGBoost
3. Save the best model
4. Start the FastAPI server at http://127.0.0.1:8000

### Step 7 — Open the dashboard
Open `frontend/index.html` in your browser (double-click the file).

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Predict single student |
| POST | `/predict/batch` | Predict list of students |
| GET | `/stats` | Dataset statistics |
| GET | `/sample-students` | 6 sample students for demo |

## API Docs
Visit http://127.0.0.1:8000/docs for interactive Swagger UI.

## Sample Request (single predict)
```json
POST /predict
{
  "student_id": "S104",
  "attendance": 58,
  "assignment_score": 46,
  "midterm_score": 42,
  "study_hours": 1.5,
  "previous_grade": 50,
  "quiz_scores": 44,
  "participation": 38,
  "sleep_hours": 5
}
```

## Sample Response
```json
{
  "student_id": "S104",
  "risk_level": "High",
  "probability": 0.82,
  "risk_score": 82.0,
  "recommendation": "Improve attendance | Schedule immediate academic counseling session"
}
```
