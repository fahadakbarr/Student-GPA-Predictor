# ── Installs (run in terminal or Colab cell) ──────────────────────────────────
# pip install xgboost catboost seaborn gradio

from IPython.utils import PyColorize
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("Imported all libraries successfully!")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\Admin\Downloads\studentgpa\Studentdata.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.columns.tolist())

# ── Drop irrelevant columns; keep GPA as target ────────────────────────────────
df.drop(['StudentID', 'GradeClass'], axis=1, inplace=True)

# ── EDA ────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.histplot(df['GPA'], bins=20, kde=True)
plt.title("Distribution of Student GPA")
plt.tight_layout()
plt.show()

mask = np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap=cmap, fmt='.2f', mask=mask)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ── Feature / target split ─────────────────────────────────────────────────────
# NOTE: all columns in this dataset are already numeric, so get_dummies is a
# no-op here, but kept for safety in case the CSV has string columns.
df = pd.get_dummies(df, drop_first=True)

X = df.drop("GPA", axis=1)
y = df["GPA"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

# ── Baseline models ────────────────────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(random_state=42),
    "XGBoost":           XGBRegressor(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\n{name}:")
    print(f"  MAE : {mean_absolute_error(y_test, preds):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
    print(f"  R²  : {r2_score(y_test, preds):.4f}")
    print("-" * 40)

# ── CatBoost ───────────────────────────────────────────────────────────────────
cat_model = CatBoostRegressor(verbose=0, random_state=42)
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)

print("\nCatBoost Results:")
print(f"  MAE : {mean_absolute_error(y_test, cat_preds):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, cat_preds)):.4f}")
print(f"  R²  : {r2_score(y_test, cat_preds):.4f}")

# ── Feature importance ─────────────────────────────────────────────────────────
print("\n" + "-" * 40)
print("IMPORTANCE OF FEATURES")
importances = cat_model.get_feature_importance(prettified=True)
display(importances)
print("-" * 40)

# ── Actual vs Predicted ────────────────────────────────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(y_test, cat_preds, alpha=0.7)
min_val = min(y_test.min(), cat_preds.min())
max_val = max(y_test.max(), cat_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("CatBoost: Actual vs Predicted GPA")
plt.tight_layout()
plt.show()

# ── Prediction helper ──────────────────────────────────────────────────────────
def predict_gpa(age, gender, ethnicity, parental_education,
                study_time_weekly, absences, tutoring,
                parental_support, extracurricular, sports, music, volunteering):
    """Predict GPA from student features using the trained CatBoost model."""
    student_features = {
        'Age':               age,
        'Gender':            gender,
        'Ethnicity':         ethnicity,
        'ParentalEducation': parental_education,
        'StudyTimeWeekly':   study_time_weekly,
        'Absences':          absences,
        'Tutoring':          tutoring,
        'ParentalSupport':   parental_support,
        'Extracurricular':   extracurricular,
        'Sports':            sports,
        'Music':             music,
        'Volunteering':      volunteering,
    }
    new_df = pd.DataFrame([student_features])
    # Align columns exactly with training data (handles any dummies created)
    new_df_processed = new_df.reindex(columns=X_train.columns, fill_value=0)
    return round(float(cat_model.predict(new_df_processed)[0]), 3)

# ── Gradio UI ──────────────────────────────────────────────────────────────────
import gradio as gr

def gradio_predict(age, gender, ethnicity, parental_education,
                   study_time, absences, tutoring,
                   parental_support, extracurricular, sports, music, volunteering):
    gpa = predict_gpa(
        age, gender, ethnicity, parental_education,
        study_time, absences, tutoring,
        parental_support, extracurricular, sports, music, volunteering
    )

    # FIX: color was defined but never used — now embedded in styled HTML badge
    if gpa >= 3.5:
        grade, color = "A  🟢", "#22c55e"
    elif gpa >= 3.0:
        grade, color = "B  🔵", "#3b82f6"
    elif gpa >= 2.0:
        grade, color = "C  🟡", "#f59e0b"
    elif gpa >= 1.0:
        grade, color = "D  🟠", "#f97316"
    else:
        grade, color = "F  🔴", "#ef4444"

    return (
        f"<div style='font-size:1.4rem; font-weight:bold; color:{color};'>"
        f"Predicted GPA: {gpa:.2f} / 4.00 &nbsp;|&nbsp; Grade: {grade}"
        f"</div>"
    )

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    title="🎓 Student GPA Predictor"
) as demo:

    gr.Markdown("# 🎓 Student GPA Predictor\nFill in the student details below and click **Predict GPA**.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Student Info")
            age                = gr.Slider(13, 20, value=17, step=1, label="Age")
            # FIX: consistent (label, value) tuple syntax for all Radio/Dropdown choices
            gender             = gr.Radio([("Female", 0), ("Male", 1)], value=0, label="Gender")
            ethnicity          = gr.Dropdown(
                                     choices=[("Caucasian", 0), ("African American", 1),
                                              ("Asian", 2), ("Other", 3)],
                                     value=0, label="Ethnicity")
            parental_education = gr.Dropdown(
                                     choices=[("None", 0), ("High School", 1),
                                              ("Some College", 2), ("Bachelor's", 3), ("Higher", 4)],
                                     value=2, label="Parental Education")

        with gr.Column():
            gr.Markdown("### 📚 Academic Habits")
            study_time       = gr.Slider(0, 40, value=15.0, step=0.5, label="Weekly Study Time (hrs)")
            absences         = gr.Slider(0, 30, value=5,    step=1,   label="Number of Absences")
            tutoring         = gr.Radio([("No", 0), ("Yes", 1)], value=0, label="Tutoring")
            parental_support = gr.Slider(0, 4,  value=2,    step=1,   label="Parental Support (0–4)")

        with gr.Column():
            gr.Markdown("### 🏅 Activities")
            extracurricular = gr.Radio([("No", 0), ("Yes", 1)], value=0, label="Extracurricular")
            sports          = gr.Radio([("No", 0), ("Yes", 1)], value=0, label="Sports")
            music           = gr.Radio([("No", 0), ("Yes", 1)], value=0, label="Music")
            volunteering    = gr.Radio([("No", 0), ("Yes", 1)], value=0, label="Volunteering")

    predict_btn = gr.Button("🔮 Predict GPA", variant="primary", size="lg")
    output      = gr.HTML(label="Result")   # FIX: use gr.HTML so the styled div renders properly

    predict_btn.click(
        fn=gradio_predict,
        inputs=[age, gender, ethnicity, parental_education,
                study_time, absences, tutoring, parental_support,
                extracurricular, sports, music, volunteering],
        outputs=output,
    )

# share=True gives a public URL in Colab; set to False for local use
demo.launch(share=True)