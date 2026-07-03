# 🎓 Student GPA Predictor

A machine-learning pipeline that predicts student GPA from demographic, academic, and extracurricular features, wrapped in an interactive **Gradio** web app.

---

## 📊 Dataset

`data/Studentdata.csv` — 2 392 student records with 15 columns:

| Column | Description |
|---|---|
| `StudentID` | Unique identifier (dropped during training) |
| `Age` | Student age (15–18) |
| `Gender` | 0 = Female, 1 = Male |
| `Ethnicity` | 0 Caucasian · 1 African American · 2 Asian · 3 Other |
| `ParentalEducation` | 0 None → 4 Higher degree |
| `StudyTimeWeekly` | Weekly study hours (continuous) |
| `Absences` | Number of school absences |
| `Tutoring` | 0 = No, 1 = Yes |
| `ParentalSupport` | 0 (None) → 4 (Very High) |
| `Extracurricular` | 0 = No, 1 = Yes |
| `Sports` | 0 = No, 1 = Yes |
| `Music` | 0 = No, 1 = Yes |
| `Volunteering` | 0 = No, 1 = Yes |
| `GPA` | Target — 0.0 to 4.0 |
| `GradeClass` | Derived grade label (dropped during training) |

---

## 🤖 Models & Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.1553 | 0.1966 | **0.9532** |
| Random Forest | 0.1908 | 0.2460 | 0.9268 |
| XGBoost | 0.1891 | 0.2398 | 0.9305 |
| **CatBoost** | **0.1668** | **0.2114** | **0.9460** |

> Linear Regression achieves the highest R² on this dataset. CatBoost is used for the interactive app because of its robust feature-importance output.

### 🔑 Top Features (CatBoost)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Absences | 66.01 % |
| 2 | StudyTimeWeekly | 10.05 % |
| 3 | ParentalSupport | 8.11 % |
| 4 | Tutoring | 4.42 % |
| 5 | Extracurricular | 2.71 % |

---

## 🚀 Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Train models & launch the app

```bash
python src/train_and_app.py
```

The Gradio UI will open at `http://127.0.0.1:7860`. Add `share=True` in the last line to get a public link.

---

## 🖥️ Gradio App

Fill in student details across three panels — **Student Info**, **Academic Habits**, and **Activities** — then click **Predict GPA**. The result shows the predicted GPA (0–4) and a colour-coded letter grade:

| GPA | Grade | Colour |
|---|---|---|
| ≥ 3.5 | A 🟢 | Green |
| ≥ 3.0 | B 🔵 | Blue |
| ≥ 2.0 | C 🟡 | Yellow |
| ≥ 1.0 | D 🟠 | Orange |
| < 1.0 | F 🔴 | Red |

---

## 📁 Project Structure

```
student-gpa-predictor/
├── data/
│   └── Studentdata.csv          # Raw dataset
├── src/
│   └── train_and_app.py         # Full pipeline + Gradio UI
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

See `requirements.txt`. Core packages:

- `pandas`, `numpy`, `scikit-learn`
- `xgboost`, `catboost`
- `matplotlib`, `seaborn`
- `gradio`

---

## Results

<img width="600" height="400" alt="Distribution of Student GPA" src="https://github.com/user-attachments/assets/21789f76-6ef7-4dfb-bc6d-009637f12f6e" />
<img width="1536" height="754" alt="Correlation Heatmap" src="https://github.com/user-attachments/assets/1ff2eb55-a0e9-4574-9a9c-e38743d35bb9" />
<img width="1536" height="754" alt="Catboost: Actual vs Predicted GPA" src="https://github.com/user-attachments/assets/9f1e9eaf-2e5b-47f4-bb1a-08e2a8773480" />

---

## Contributors
- Fahad Akbar
- Abdullah Baig

