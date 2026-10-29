"""
preprocess.py
-------------
Repeatable data loading, cleaning, missingness analysis, and discretisation
pipeline for the UCI Heart Disease dataset.

Key MLOps Principles:
- Strict prevention of data leakage: continuous medians and categorical modes
  are computed exclusively from the training split.
- Deterministic binning based on clinically grounded cutoffs.
- Detailed missingness report generated and saved for auditability.
- Dataset MD5 hash calculated for provenance tracking.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Raw and target definitions
RAW_FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "heart_disease"

DISCRETIZED_COLS = [
    "age_bin", "sex", "cp", "trestbps_bin", "chol_bin",
    "fbs", "restecg", "thalch_bin", "exang", "oldpeak_bin",
    "slope", "ca", "thal"
]


def compute_file_md5(filepath: str) -> str:
    """Calculate MD5 checksum for dataset provenance tracking."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {filepath}")
    df = pd.read_csv(path)
    logger.info(f"Loaded raw dataset from {filepath} with shape {df.shape}")
    return df


def generate_missing_data_report(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Generate an in-depth audit of dataset missingness, including per-hospital breakdown
    (Cleveland, Hungary, Switzerland, VA Long Beach) and clinical implications.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100

    report_lines = [
        "=" * 70,
        "DATASET AUDIT & MISSING DATA REPORT",
        "UCI Heart Disease Comprehensive Multi-Center Dataset",
        "=" * 70,
        f"Total Instances: {total_rows}",
        f"Total Attributes: {df.shape[1]}",
        "",
        "1. Overall Column Missingness Profile:",
        "-" * 70,
        f"{'Column':<15} {'Missing Count':<15} {'Missing (%)':<15} {'Data Type'}",
        "-" * 70
    ]

    for col in df.columns:
        report_lines.append(
            f"{col:<15} {missing_counts[col]:<15} {missing_pct[col]:<14.2f}% {str(df[col].dtype)}"
        )

    # Multi-center breakdown
    if "dataset" in df.columns:
        report_lines.extend([
            "",
            "2. Missingness Analysis Across Hospital Centers:",
            "-" * 70
        ])
        center_summary = df.groupby("dataset").apply(
            lambda g: pd.Series({
                "Total Rows": len(g),
                "Missing ca": g["ca"].isnull().sum() if "ca" in g else 0,
                "Missing thal": g["thal"].isnull().sum() if "thal" in g else 0,
                "Missing slope": g["slope"].isnull().sum() if "slope" in g else 0,
                "Zero/Missing chol": ((g["chol"] == 0) | (g["chol"].isnull())).sum() if "chol" in g else 0,
            }),
            include_groups=False
        )
        report_lines.append(center_summary.to_string())

    report_lines.extend([
        "",
        "3. Clinical Insights & Domain Assumptions:",
        "-" * 70,
        "- 'ca' (fluoroscopy) and 'thal' (thallium stress test) show heavy missingness",
        "  (66.4% and 52.8%) outside the Cleveland clinic because non-US and VA centers",
        "  did not routinely perform these invasive or radio-isotope diagnostic procedures.",
        "- 'chol' has 172 zeros (primarily in VA Long Beach) where 0 denotes unmeasured cholesterol.",
        "- 'trestbps' has 1 zero entry representing unrecorded resting blood pressure.",
        "- In this pipeline, zeros in trestbps/chol are treated as NaN and imputed via",
        "  training-set medians. Missing categorical test results are imputed via training-set modes",
        "  to retain all 920 records without data leakage.",
        "=" * 70
    ])

    report_text = "\n".join(report_lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"Missing data report written to {output_path}")

    return report_text


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize raw columns:
    - Binarize 'num' into 'heart_disease' (0 = absence, 1 = presence: stages 1, 2, 3, 4)
    - Treat physiological zeros in trestbps and chol as missing values
    - Clean boolean string representations
    """
    data = df.copy()

    # Binarize target
    if "num" in data.columns:
        data[TARGET_COL] = (data["num"] > 0).astype(int)
        data = data.drop(columns=["num"])

    # Drop administrative non-clinical identifier
    if "id" in data.columns:
        data = data.drop(columns=["id"])

    # Physiological zero correction
    if "trestbps" in data.columns:
        data.loc[data["trestbps"] <= 0, "trestbps"] = np.nan
    if "chol" in data.columns:
        data.loc[data["chol"] <= 0, "chol"] = np.nan

    # Drop instances where target itself might be NaN
    data = data.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    return data


def compute_train_statistics(train_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute imputers strictly from the training split to prevent data leakage:
    - Median for continuous variables
    - Mode for categorical variables
    """
    continuous_cols = ["trestbps", "chol", "thalch", "oldpeak"]
    categorical_cols = ["fbs", "restecg", "exang", "slope", "ca", "thal"]

    medians = {}
    for col in continuous_cols:
        med = float(train_df[col].dropna().median())
        medians[col] = med

    modes = {}
    for col in categorical_cols:
        mode_val = train_df[col].dropna().mode()
        modes[col] = str(mode_val[0]) if len(mode_val) > 0 else "unknown"

    return {"medians": medians, "modes": modes}


def discretize_dataset(
    df: pd.DataFrame,
    stats: Dict[str, Any]
) -> pd.DataFrame:
    """
    Discretize continuous variables into clinically validated ordinal bins
    and impute missing values using training statistics.
    """
    d = df.copy()
    medians = stats["medians"]
    modes = stats["modes"]

    # 1. Age: young (<45), middle (45-60), senior (>60)
    d["age_bin"] = pd.cut(
        d["age"],
        bins=[0, 45, 60, 120],
        labels=["young", "middle", "senior"],
        include_lowest=True
    ).astype(str)

    # 2. Resting Blood Pressure: normal (<120), elevated (120-140), high (>140)
    trestbps_val = d["trestbps"].fillna(medians["trestbps"])
    d["trestbps_bin"] = pd.cut(
        trestbps_val,
        bins=[0, 120, 140, 300],
        labels=["normal", "elevated", "high"],
        include_lowest=True
    ).astype(str)

    # 3. Cholesterol: normal (<200), borderline (200-240), high (>240)
    chol_val = d["chol"].fillna(medians["chol"])
    d["chol_bin"] = pd.cut(
        chol_val,
        bins=[0, 200, 240, 1000],
        labels=["normal", "borderline", "high"],
        include_lowest=True
    ).astype(str)

    # 4. Maximum Heart Rate (thalch): low (<120), moderate (120-150), high (>150)
    thalch_val = d["thalch"].fillna(medians["thalch"])
    d["thalch_bin"] = pd.cut(
        thalch_val,
        bins=[0, 120, 150, 300],
        labels=["low", "moderate", "high"],
        include_lowest=True
    ).astype(str)

    # 5. ST Depression (oldpeak): none (<=0), mild (0-1.5), severe (>1.5)
    oldpeak_val = d["oldpeak"].fillna(medians["oldpeak"])
    d["oldpeak_bin"] = pd.cut(
        oldpeak_val,
        bins=[-10.0, 0.0, 1.5, 20.0],
        labels=["none", "mild", "severe"],
        include_lowest=True
    ).astype(str)

    # 6. Categorical features imputed via training modes
    for col, mode_val in modes.items():
        if col in d.columns:
            d[col] = d[col].astype(object).fillna(mode_val).astype(str)

    # 7. Standard categorical features
    d["sex"] = d["sex"].astype(str)
    d["cp"] = d["cp"].astype(str)

    # Select final modeling columns
    output_cols = DISCRETIZED_COLS + [TARGET_COL]
    final_df = d[output_cols].copy()
    for col in output_cols:
        final_df[col] = final_df[col].astype(str)

    return final_df


def run_preprocessing_pipeline(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    missing_report_path: str = "reports/missing_data_report.txt"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end preprocessing pipeline:
    1. Load raw data and compute provenance hash
    2. Audit missing data and write report
    3. Clean raw data (physiological zeros, binarize target)
    4. Stratified split into train and test sets
    5. Compute imputers exclusively on train set (zero data leakage)
    6. Discretize both splits
    """
    raw_df = load_raw_data(data_path)
    md5_hash = compute_file_md5(data_path)

    # Audit & report
    generate_missing_data_report(raw_df, missing_report_path)

    # Clean raw data
    cleaned_df = clean_raw_data(raw_df)

    # Stratified split to prevent leakage
    train_raw, test_raw = train_test_split(
        cleaned_df,
        test_size=test_size,
        random_state=random_state,
        stratify=cleaned_df[TARGET_COL]
    )

    # Compute statistics strictly on train
    stats = compute_train_statistics(train_raw)

    # Discretize using train stats
    train_df = discretize_dataset(train_raw, stats)
    test_df = discretize_dataset(test_raw, stats)

    # Collect complete pipeline metadata
    metadata = {
        "source_data_path": os.path.abspath(data_path),
        "data_md5": md5_hash,
        "raw_total_rows": len(raw_df),
        "cleaned_total_rows": len(cleaned_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_size": test_size,
        "random_state": random_state,
        "train_statistics": stats,
        "discretized_features": DISCRETIZED_COLS,
        "target_col": TARGET_COL,
        "target_distribution_train": train_df[TARGET_COL].value_counts().to_dict(),
        "target_distribution_test": test_df[TARGET_COL].value_counts().to_dict(),
    }

    logger.info(
        f"Preprocessing complete. Train: {len(train_df)} rows, Test: {len(test_df)} rows. "
        f"Dataset MD5: {md5_hash}"
    )

    return train_df, test_df, metadata


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess Heart Disease dataset")
    parser.add_argument("--data_path", default="data/heart_disease_uci.csv")
    parser.add_argument("--report_path", default="reports/missing_data_report.txt")
    args = parser.parse_args()

    train_df, test_df, meta = run_preprocessing_pipeline(
        args.data_path, missing_report_path=args.report_path
    )
    print("Preprocess run successful!")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
