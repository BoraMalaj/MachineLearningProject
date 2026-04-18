"""
Phase 1  Preprocessing
===========================================

This module implements the preprocessing pipeline for the healthcare stroke
dataset. It is intentionally written as a thin, import-safe module so that
Phase 2 (modelling) can re-use the exact same transformer by calling
``build_preprocessor``.

Key design choices
------------------
* Leakage prevention: all statistics (imputation medians, scaler means/stds,
  one-hot categories) are fitted on the training split only and then applied
  to the test split via an sklearn ``ColumnTransformer``.
* Stratified 80/20 split with ``random_state=42`` to preserve the severe
  class imbalance (~4.9% positive).
* ``bmi`` missing values (encoded as the string "N/A" in the raw CSV) are
  imputed with the training-set median. KNN and iterative imputation were
  considered; median was retained as it is robust to the right-skew of BMI
  and deterministic.
* ``smoking_status == "Unknown"`` is preserved as its own category rather
  than imputed, because "Unknown" carries signal and represents roughly a
  third of the observations.
* One-hot encoding with ``handle_unknown='ignore'`` and ``drop='if_binary'``
  to keep binary nominal variables as a single column.
* ``StandardScaler`` for numeric features to support distance- and
  gradient-based learners (Logistic Regression, SVM).
* Class imbalance is not addressed inside preprocessing; a SMOTE-augmented
  training set is saved as an optional second artefact so Phase 2 can
  choose the resampling strategy.

Running this script directly (``python -m src.preprocessing`` or
``python src/preprocessing.py``) will produce all preprocessing artefacts
under ``outputs/``.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    IMBLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_FEATURES = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
TARGET = "stroke"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply dataset-level cleaning.

    Steps:
        1. Read the CSV.
        2. Replace the string ``"N/A"`` in ``bmi`` with ``NaN`` and cast
           to float.
        3. Drop the ``id`` column (identifier, not a feature).
        4. Drop the single ``gender == "Other"`` row, which is insufficient
           to model and would create a tiny one-hot dimension.
    """
    df = pd.read_csv(csv_path)

    # bmi is read as object because of the "N/A" sentinel.
    df["bmi"] = df["bmi"].replace("N/A", np.nan).astype(float)

    # Drop identifier column.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Drop the single "Other" gender record.
    n_other = int((df["gender"] == "Other").sum())
    if n_other:
        df = df.loc[df["gender"] != "Other"].reset_index(drop=True)

    return df


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Return an unfitted ``ColumnTransformer`` implementing the pipeline.

    The numeric branch imputes missing values with the median and then
    standardises the result. The categorical branch imputes with the most
    frequent value (defensive - the raw training set has no missing
    categoricals, but this protects against unseen inputs at serving time)
    and one-hot encodes.

    Parameters
    ----------
    numeric_features, categorical_features
        Optional overrides; default to the module-level lists.
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split on the binary target."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def fit_transform_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Fit the preprocessor on the training split and transform both.

    Returns processed ``DataFrame`` objects carrying feature names recovered
    via ``get_feature_names_out``.
    """
    preprocessor = build_preprocessor()
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_processed = pd.DataFrame(
        X_train_arr, columns=feature_names, index=X_train.index
    )
    X_test_processed = pd.DataFrame(
        X_test_arr, columns=feature_names, index=X_test.index
    )
    return X_train_processed, X_test_processed, preprocessor


def apply_smote(
    X_train_processed: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
):
    """Apply SMOTE to the (already preprocessed) training data.

    Returns ``(X_train_smote, y_train_smote)`` as DataFrame/Series, or
    ``(None, None)`` when ``imblearn`` is unavailable.
    """
    if not IMBLEARN_AVAILABLE:
        return None, None

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train_processed, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train_processed.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    return X_res, y_res


def save_artifacts(
    X_train_processed: pd.DataFrame,
    X_test_processed: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    X_train_smote: pd.DataFrame | None,
    y_train_smote: pd.Series | None,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """Persist all preprocessing artefacts to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    def _write_csv(obj: pd.DataFrame | pd.Series, name: str) -> None:
        path = output_dir / name
        obj.to_csv(path, index=False)
        written.append(path)

    _write_csv(X_train_processed, "X_train.csv")
    _write_csv(X_test_processed, "X_test.csv")
    _write_csv(y_train.to_frame(), "y_train.csv")
    _write_csv(y_test.to_frame(), "y_test.csv")

    if X_train_smote is not None and y_train_smote is not None:
        _write_csv(X_train_smote, "X_train_smote.csv")
        _write_csv(y_train_smote.to_frame(), "y_train_smote.csv")

    preproc_path = output_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preproc_path)
    written.append(preproc_path)

    return written


# ---------------------------------------------------------------------------
# Script entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading dataset from: {DATA_PATH}")
    df = load_and_clean(DATA_PATH)
    print(f"Cleaned dataset shape: {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df)
    print(
        f"Train shape: {X_train.shape} | Test shape: {X_test.shape}\n"
        f"Train class distribution:\n{y_train.value_counts(normalize=True)}\n"
        f"Test class distribution:\n{y_test.value_counts(normalize=True)}"
    )

    X_train_processed, X_test_processed, preprocessor = fit_transform_split(
        X_train, X_test
    )
    print(
        f"Processed train matrix: {X_train_processed.shape}, "
        f"processed test matrix: {X_test_processed.shape}"
    )

    if IMBLEARN_AVAILABLE:
        X_train_smote, y_train_smote = apply_smote(X_train_processed, y_train)
        print(
            "SMOTE applied. Before:\n"
            f"{y_train.value_counts()}\nAfter:\n{y_train_smote.value_counts()}"
        )
    else:
        X_train_smote, y_train_smote = None, None
        print(
            "imblearn not installed; skipping SMOTE artefact. Install "
            "`imbalanced-learn` to enable."
        )

    written = save_artifacts(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        X_train_smote,
        y_train_smote,
    )
    print("Saved artefacts:")
    for path in written:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
