# Stroke Prediction — Student A, Phase 1

Machine Learning classification project on the Kaggle **Stroke Prediction Dataset**
(`fedesoriano`, derived from public electronic health records). This repository
hosts the **Phase 1** deliverables of Student A: problem definition, exploratory
data analysis, preprocessing, and feature engineering. Phase 2 (modelling) will
be contributed by Students B (SVM) and C (XGBoost / gradient boosting) and will
re-use the train/test split and the fitted `ColumnTransformer` persisted here.

## Dataset

- **File**: `data/healthcare-dataset-stroke-data.csv`
- **Rows**: 5,110 patient records
- **Columns**: 12 (`id`, `gender`, `age`, `hypertension`, `heart_disease`,
  `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`,
  `smoking_status`, `stroke`)
- **Target**: `stroke` (binary; approximately 4.9% positive — severe class
  imbalance, roughly 19:1)
- **Known data quirks**:
  - `bmi` is stored as an object column because missing values are encoded as
    the literal string `"N/A"` (≈3.9% of rows).
  - `smoking_status` contains an `"Unknown"` category that is kept as its own
    level rather than imputed.
  - There is exactly one row with `gender == "Other"`; it is dropped because it
    is insufficient to model and creates a degenerate one-hot dimension.

## Repository layout

```
.
├── data/
│   └── healthcare-dataset-stroke-data.csv   # Raw dataset
├── notebooks/
│   ├── 01_problem_definition.ipynb          # Motivation, formulation, schema
│   ├── 02_eda.ipynb                         # Univariate / bivariate / multivariate
│   ├── 03_preprocessing.ipynb               # Split, impute, scale, encode, SMOTE
│   └── 04_feature_engineering.ipynb         # Clinical features, MI, selection
├── src/
│   ├── __init__.py
│   ├── problem_definition.py                # Headless mirror of notebook 01
│   ├── eda.py                               # Headless mirror of notebook 02
│   ├── preprocessing.py                     # Importable pipeline for Phase 2
│   └── feature_engineering.py               # `add_clinical_features(df)` helper
├── figures/                                 # Produced by EDA / FE scripts
├── outputs/                                 # Saved splits + joblib preprocessor
├── requirements.txt
└── README.md
```

## Phase 1 scope (Student A)

| Section                       | Notebook                         | Script                          |
| ----------------------------- | -------------------------------- | ------------------------------- |
| Problem definition            | `01_problem_definition.ipynb`    | `src/problem_definition.py`     |
| Exploratory data analysis     | `02_eda.ipynb`                   | `src/eda.py`                    |
| Preprocessing pipeline        | `03_preprocessing.ipynb`         | `src/preprocessing.py`          |
| Feature engineering           | `04_feature_engineering.ipynb`   | `src/feature_engineering.py`    |

The notebooks are the primary deliverable and contain the academic narrative
required for the conference-format paper. The Python scripts are headless
mirrors: they produce the same figures and artefacts but can be run from the
command line or imported as modules from Phase 2 code.

## Evaluation strategy (shared across Students A, B, C)

- Stratified 80 / 20 train / test split with `random_state=42`.
- 5-fold stratified cross-validation for hyperparameter search.
- **Primary metric**: ROC-AUC.
- **Secondary metrics**: PR-AUC, recall, F1, balanced accuracy, Brier score.
- The fitted `ColumnTransformer` is persisted to
  `outputs/preprocessor.joblib` so Phase 2 models fit identical inputs.

## How to run

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the notebooks

```bash
jupyter lab
# or
jupyter notebook
```

Execute them in order: `01 → 02 → 03 → 04`. Notebook 03 writes the train/test
splits and the fitted preprocessor to `outputs/`, which notebook 04 and all
Phase 2 notebooks consume.

### 3. Run the headless scripts (optional)

```bash
python src/problem_definition.py
python src/eda.py
python src/preprocessing.py
python src/feature_engineering.py
```

All scripts use `pathlib` and resolve paths relative to the repository root,
so they can be invoked from any working directory.

## Key design decisions

1. **Leakage prevention.** Every preprocessing statistic (imputation medians,
   scaler means and standard deviations, one-hot categories) is fitted on the
   training split only and applied to the test split through an
   `sklearn.compose.ColumnTransformer`.
2. **Missing BMI.** Median imputation is used by default because BMI is
   right-skewed. KNN and iterative imputation are discussed as alternatives in
   the preprocessing notebook.
3. **Smoking status.** The `"Unknown"` level is preserved as its own category,
   because removing or imputing it discards the information that the status
   was not recorded.
4. **Class imbalance.** Phase 1 does **not** commit to a resampling strategy.
   Instead, a SMOTE-augmented training set is saved alongside the raw split so
   Phase 2 can compare `class_weight='balanced'`, SMOTE, and the unmodified
   training data on equal footing.
5. **Feature engineering.** Clinically motivated features (age bands, WHO BMI
   categories, ADA glucose categories, comorbidity count, metabolic-risk flags,
   age × glucose interaction, log-glucose, age²) are added as raw columns in
   `stroke_engineered.csv`. Their statistical relevance is validated with
   mutual information and chi-squared tests.

## Artefacts produced

Running the preprocessing notebook / script writes to `outputs/`:

- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` — processed splits
- `X_train_smote.csv`, `y_train_smote.csv` — SMOTE-augmented training set
  (only if `imbalanced-learn` is installed)
- `preprocessor.joblib` — the fitted `ColumnTransformer`

Running the feature engineering notebook / script writes:

- `outputs/stroke_engineered.csv` — the raw dataset plus engineered columns

Running the EDA notebook / script writes approximately nine figures to
`figures/` (missingness, univariate distributions, bivariate plots with the
target, correlation heatmap, outlier boxplots, multivariate scatter plots).

## Reproducibility

- All stochastic steps use `random_state=42`.
- The fitted preprocessor is persisted with `joblib`.
- `requirements.txt` pins the core scientific stack.
- The notebooks have cleared outputs (`execution_count: null`) so diffs stay
  small in version control.

## AI and plagiarism disclosure

This repository contains code and prose drafted with the assistance of large
language models. Every statistic reported in the notebooks is computed
directly from the CSV; no numerical values were hand-written. The final paper
includes the explicit AI-usage appendix required by the course rubric.
