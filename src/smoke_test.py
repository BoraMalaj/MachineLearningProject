"""Quick smoke test: verify every Phase 1 artifact exists and is loadable."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    marker = "PASS" if condition else "FAIL"
    line = f"  [{marker}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not condition:
        failures.append(label)


print("=" * 70)
print("Phase 1 smoke test")
print("=" * 70)

print("\n[1/5] Notebooks present and non-trivial")
for nb in [
    "01_problem_definition.ipynb",
    "02_eda.ipynb",
    "03_preprocessing.ipynb",
    "04_feature_engineering.ipynb",
]:
    p = NOTEBOOKS / nb
    size_kb = p.stat().st_size // 1024 if p.exists() else 0
    check(nb, p.exists() and size_kb > 10, f"{size_kb} KB")

print("\n[2/5] Figures produced")
expected_figures = [
    "01_class_distribution.png",
    "02_missingness.png",
    "02_numeric_distributions.png",
    "02_categorical_distributions.png",
    "02_target_distribution.png",
    "02_numeric_vs_target.png",
    "02_categorical_vs_target.png",
    "02_correlation_heatmap.png",
    "02_outliers.png",
    "02_multivariate.png",
    "04_engineered_stroke_rates.png",
    "04_feature_importance_mi.png",
    "04_engineered_correlations.png",
]
for fig in expected_figures:
    p = FIGURES / fig
    check(fig, p.exists() and p.stat().st_size > 1000)

print("\n[3/5] Preprocessing artifacts loadable")
X_train = pd.read_csv(OUTPUTS / "X_train.csv")
X_test = pd.read_csv(OUTPUTS / "X_test.csv")
y_train = pd.read_csv(OUTPUTS / "y_train.csv")
y_test = pd.read_csv(OUTPUTS / "y_test.csv")
check("X_train shape", X_train.shape[0] == 4087 and X_train.shape[1] == 17,
      f"{X_train.shape}")
check("X_test shape", X_test.shape[0] == 1022 and X_test.shape[1] == 17,
      f"{X_test.shape}")
check("Stratification preserved on train",
      abs(y_train["stroke"].mean() - 0.0487) < 0.005,
      f"positive rate = {y_train['stroke'].mean():.4f}")
check("Stratification preserved on test",
      abs(y_test["stroke"].mean() - 0.0487) < 0.005,
      f"positive rate = {y_test['stroke'].mean():.4f}")

print("\n[4/5] SMOTE artifacts balanced")
X_smote = pd.read_csv(OUTPUTS / "X_train_smote.csv")
y_smote = pd.read_csv(OUTPUTS / "y_train_smote.csv")
counts = y_smote["stroke"].value_counts().to_dict()
check("SMOTE produced balanced classes",
      counts.get(0) == counts.get(1),
      f"class counts = {counts}")

print("\n[5/5] Fitted preprocessor and engineered dataset")
preprocessor = joblib.load(OUTPUTS / "preprocessor.joblib")
check("preprocessor.joblib loads", preprocessor is not None,
      type(preprocessor).__name__)

eng = pd.read_csv(OUTPUTS / "stroke_engineered.csv")
expected_engineered = [
    "age_group", "bmi_category", "glucose_category",
    "comorbidity_count", "smoking_risk", "is_metabolic_risk",
    "is_high_risk_demographic", "age_glucose", "bmi_age",
    "log_glucose", "age_squared",
]
missing = [c for c in expected_engineered if c not in eng.columns]
check("all 11 engineered features present",
      len(missing) == 0,
      f"shape = {eng.shape}")
if missing:
    print(f"    missing: {missing}")

print("\n" + "=" * 70)
if failures:
    print(f"SMOKE TEST FAILED  ({len(failures)} failures)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("SMOKE TEST PASSED  (all artifacts present and valid)")
    sys.exit(0)
