"""
Phase 1 - Student A - Part 1: Problem Definition and Dataset Understanding.

Headless mirror of notebooks/01_problem_definition.ipynb. Loads the stroke
prediction dataset, prints a structural summary, computes the class
distribution and missingness, and saves a class-distribution bar chart to
the figures directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Resolve project paths relative to this file so the script works regardless
# of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "healthcare-dataset-stroke-data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURE_PATH = FIGURES_DIR / "01_class_distribution.png"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the stroke dataset, interpreting the literal string 'N/A' as NaN."""
    return pd.read_csv(path, na_values=["N/A"])


def describe_structure(df: pd.DataFrame) -> None:
    """Print shape, dtypes, head, and info for a high-level structural view."""
    print("=" * 72)
    print("Dataset structural summary")
    print("=" * 72)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head())
    print("\nInfo:")
    df.info()


def summarize_target(df: pd.DataFrame, target: str = "stroke") -> None:
    """Print class counts, percentages, and the imbalance ratio."""
    counts = df[target].value_counts().sort_index()
    percentages = df[target].value_counts(normalize=True).sort_index() * 100.0
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    imbalance_ratio = (negatives / positives) if positives > 0 else float("nan")

    print("\n" + "=" * 72)
    print("Target distribution")
    print("=" * 72)
    print("Counts:")
    print(counts)
    print("\nPercentages (%):")
    print(percentages.round(4))
    print(f"\nNegative : Positive = {negatives} : {positives}")
    print(f"Imbalance ratio (neg / pos) = {imbalance_ratio:.2f}")


def summarize_missing_and_categorical(df: pd.DataFrame) -> None:
    """Print missingness per column and unique values for categorical columns."""
    print("\n" + "=" * 72)
    print("Missing values per column")
    print("=" * 72)
    print(df.isna().sum())

    categorical_columns = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    print("\n" + "=" * 72)
    print("Unique values per categorical column")
    print("=" * 72)
    for column in categorical_columns:
        unique_values = sorted(df[column].dropna().unique().tolist())
        print(f"{column} ({len(unique_values)}): {unique_values}")


def plot_class_distribution(
    df: pd.DataFrame,
    output_path: Path,
    target: str = "stroke",
) -> None:
    """Save a bar chart of the target class distribution."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df[target].value_counts().sort_index()
    labels = ["No stroke (0)", "Stroke (1)"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_title("Class distribution of the target variable 'stroke'")
    ax.set_ylabel("Number of patients")
    ax.set_xlabel("Class")

    total = int(counts.sum())
    for bar, value in zip(bars, counts.values):
        percentage = 100.0 * value / total
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(value)}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved class distribution figure to: {output_path}")


def main() -> None:
    df = load_dataset(DATA_PATH)
    describe_structure(df)
    summarize_target(df)
    summarize_missing_and_categorical(df)
    plot_class_distribution(df, FIGURE_PATH)


if __name__ == "__main__":
    main()
