"""
train.py
--------
Training pipeline for the Bayesian Network Medical Diagnosis Model.

Responsibilities:
1. Orchestrate repeatable preprocessing and data loading.
2. Define the Bayesian Network DAG structure and conditional independence assumptions.
3. Fit Conditional Probability Tables (CPTs) via Discrete MLE.
4. Save and version all model artifacts:
   - Serialized model: `models/bayesian_network.pkl`
   - Versioned CPTs: `models/cpt_tables.json`
   - Preprocessor metadata: `models/preprocessor.json`
   - Training data snapshot: `models/data_snapshot.csv`
   - Full model provenance metadata: `models/model_meta.json`
5. Track experiments in MLflow (parameters, metrics, artifacts, dataset provenance).
"""

import os
import sys
import json
import pickle
import shutil
import logging
import datetime
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import mlflow
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import run_preprocessing_pipeline, DISCRETIZED_COLS, TARGET_COL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_git_commit_hash() -> str:
    """Retrieve current Git commit SHA for reproducibility, or fallback."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "git-commit-unavailable"


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML project configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_network_structure(features: list, target: str) -> DiscreteBayesianNetwork:
    """
    Construct the Bayesian Network Directed Acyclic Graph (DAG).

    Network Assumptions & Rationale:
    1. Structure: Diagnostic Naive Bayes topology where the target condition
       (heart_disease) is the parent of observed demographic, clinical, and
       diagnostic findings:
         heart_disease -> [age_bin, sex, cp, trestbps_bin, chol_bin,
                           fbs, restecg, thalch_bin, exang, oldpeak_bin,
                           slope, ca, thal]
    2. Clinical Basis: The pathological state (coronary artery disease) causally
       influences chest pain symptoms (cp), ischemic ECG changes (restecg, slope, oldpeak),
       fluoroscopic vessel calcification (ca), impaired perfusion (thal), and maximum
       heart rate attainment (thalch).
    3. Conditional Independence: Given the presence or absence of heart disease,
       the evidence variables are assumed conditionally independent:
         P(X_1, ..., X_n | heart_disease) = PROD_i P(X_i | heart_disease)
       This assumption enables robust, tractable parameter estimation from moderate-sized
       cohorts (920 cases) without exponential parameter explosion, while retaining high
       diagnostic discrimination and exact posterior computation for partial evidence.
    """
    edges = [(target, feat) for feat in features]
    model = DiscreteBayesianNetwork(edges)
    logger.info(f"Initialized DiscreteBayesianNetwork with {len(edges)} directed edges from '{target}'")
    return model


def export_cpt_tables(model: DiscreteBayesianNetwork) -> Dict[str, Any]:
    """Export all learned Conditional Probability Distributions into JSON format."""
    cpt_dict = {}
    for cpd in model.get_cpds():
        cpt_dict[cpd.variable] = {
            "variable": cpd.variable,
            "variable_cardinality": int(cpd.variable_card),
            "evidence_nodes": list(cpd.variables[1:]),
            "probabilities": cpd.values.tolist(),
            "state_names": {k: list(v) for k, v in cpd.state_names.items()}
        }
    return cpt_dict


def train_model(
    config_path: str = "config.yaml"
) -> Tuple[DiscreteBayesianNetwork, Dict[str, Any]]:
    """
    Execute full training pipeline with artifact generation and MLflow logging.
    """
    config = load_config(config_path)

    raw_data_path = config["paths"]["raw_data"]
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["reports_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

    # 1. Preprocessing and data loading
    logger.info("Executing preprocessing and missing data audit...")
    train_df, test_df, prep_meta = run_preprocessing_pipeline(
        data_path=raw_data_path,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        missing_report_path=config["paths"]["missing_data_report"]
    )

    # 2. Build DAG
    model = build_network_structure(
        features=config["data"]["discretized_features"],
        target=config["data"]["target_col"]
    )

    # 3. Fit CPTs
    logger.info("Fitting Bayesian Network parameters using Discrete MLE...")
    model.fit(train_df)

    # Validate model DAG and CPD sum-to-one constraints
    is_valid = model.check_model()
    logger.info(f"Model consistency check: {is_valid}")
    if not is_valid:
        raise ValueError("Constructed Bayesian Network failed consistency validation!")

    # 4. Evaluate quickly on training split for baseline metrics
    infer = VariableElimination(model)
    features = config["data"]["discretized_features"]
    target = config["data"]["target_col"]

    logger.info("Computing training set baseline performance...")
    train_preds, train_probs = [], []
    for _, row in train_df.iterrows():
        ev = {feat: row[feat] for feat in features}
        q = infer.query(variables=[target], evidence=ev, show_progress=False)
        states = list(q.state_names[target])
        p1 = float(q.values[states.index("1")])
        train_probs.append(p1)
        train_preds.append(1 if p1 >= 0.5 else 0)

    y_train = train_df[target].astype(int).values
    train_acc = float(accuracy_score(y_train, train_preds))
    train_f1 = float(f1_score(y_train, train_preds))
    train_auc = float(roc_auc_score(y_train, train_probs))

    logger.info(
        f"Train Baseline - Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, ROC-AUC: {train_auc:.4f}"
    )

    # 5. Extract and export CPTs
    cpt_export = export_cpt_tables(model)
    cpt_path = config["paths"]["cpt_tables"]
    with open(cpt_path, "w", encoding="utf-8") as f:
        json.dump(cpt_export, f, indent=2)
    logger.info(f"Exported versioned CPT tables to {cpt_path}")

    # 6. Save preprocessor metadata
    prep_meta_path = model_dir / "preprocessor.json"
    with open(prep_meta_path, "w", encoding="utf-8") as f:
        json.dump(prep_meta, f, indent=2)

    # 7. Save serialized model artifact
    model_artifact_path = config["paths"]["model_artifact"]
    with open(model_artifact_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved model binary to {model_artifact_path}")

    # 8. Save data snapshot for full reproducibility and data provenance
    snapshot_path = config["paths"]["data_snapshot"]
    shutil.copyfile(raw_data_path, snapshot_path)
    logger.info(f"Saved dataset snapshot to {snapshot_path}")

    # Save processed train/test splits for evaluate.py
    train_df.to_csv(model_dir / "train_processed.csv", index=False)
    test_df.to_csv(model_dir / "test_processed.csv", index=False)

    # 9. Track in MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_uri = config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    git_hash = get_git_commit_hash()
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Parameters
        mlflow.log_param("project_name", config["project"]["name"])
        mlflow.log_param("model_version", config["project"]["version"])
        mlflow.log_param("network_type", config["model"]["network_type"])
        mlflow.log_param("estimator", config["model"]["estimator"])
        mlflow.log_param("dataset_md5", prep_meta["data_md5"])
        mlflow.log_param("git_commit", git_hash)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("num_features", len(features))
        mlflow.log_param("edges_count", len(model.edges()))

        # Metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("train_roc_auc", train_auc)

        # Artifacts
        mlflow.log_artifact(model_artifact_path)
        mlflow.log_artifact(cpt_path)
        mlflow.log_artifact(str(prep_meta_path))
        mlflow.log_artifact(snapshot_path)
        mlflow.log_artifact(config["paths"]["missing_data_report"])

    # 10. Write model metadata JSON
    model_metadata = {
        "project": config["project"]["name"],
        "version": config["project"]["version"],
        "trained_at_utc": run_timestamp,
        "git_commit": git_hash,
        "mlflow_run_id": run_id,
        "data_provenance_md5": prep_meta["data_md5"],
        "raw_data_file": raw_data_path,
        "training_rows": len(train_df),
        "validation_rows": len(test_df),
        "target_variable": target,
        "nodes": list(model.nodes()),
        "edges": [list(edge) for edge in model.edges()],
        "train_metrics": {
            "accuracy": train_acc,
            "f1": train_f1,
            "roc_auc": train_auc
        },
        "model_artifact": model_artifact_path,
        "cpt_artifact": cpt_path,
        "preprocessor_artifact": str(prep_meta_path),
        "data_snapshot": snapshot_path
    }

    meta_path = config["paths"]["model_metadata"]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"Saved complete model metadata to {meta_path}")

    return model, model_metadata


if __name__ == "__main__":
    train_model()
