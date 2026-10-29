"""
test_pipeline.py
----------------
Comprehensive test suite verifying data preprocessing, Bayesian network consistency,
probabilistic inference with partial evidence, and FastAPI endpoints.
Can be executed with standard Python: `python tests/test_pipeline.py`.
"""

import os
import sys
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import run_preprocessing_pipeline, load_raw_data, clean_raw_data
from src.inference import BayesianDiagnosisInference
from fastapi.testclient import TestClient
from app.api import app


class TestBayesianPipeline(unittest.TestCase):

    def test_01_raw_data_loading(self):
        """Verify raw dataset exists and can be loaded."""
        data_path = PROJECT_ROOT / "data" / "heart_disease_uci.csv"
        self.assertTrue(data_path.exists(), "Raw dataset CSV missing")
        df = load_raw_data(str(data_path))
        self.assertEqual(len(df), 920, f"Expected 920 rows, got {len(df)}")
        self.assertIn("num", df.columns)

    def test_02_preprocessing_and_leakage_prevention(self):
        """Verify that preprocessing splits data and prevents leakage."""
        data_path = str(PROJECT_ROOT / "data" / "heart_disease_uci.csv")
        train_df, test_df, meta = run_preprocessing_pipeline(
            data_path, test_size=0.2, random_state=42
        )

        self.assertEqual(len(train_df), 736)
        self.assertEqual(len(test_df), 184)
        self.assertEqual(train_df.isnull().sum().sum(), 0, "Train dataset contains NaNs")
        self.assertEqual(test_df.isnull().sum().sum(), 0, "Test dataset contains NaNs")
        self.assertIsNotNone(meta["data_md5"])

    def test_03_cpt_mathematical_consistency(self):
        """Verify that all CPTs in the saved model sum to 1.0 (valid probability distributions)."""
        cpt_file = PROJECT_ROOT / "models" / "cpt_tables.json"
        self.assertTrue(cpt_file.exists(), "cpt_tables.json missing")
        with open(cpt_file, "r", encoding="utf-8") as f:
            cpts = json.load(f)

        for var, cpd in cpts.items():
            probs = np.array(cpd["probabilities"])
            col_sums = probs.sum(axis=0)
            self.assertTrue(np.allclose(col_sums, 1.0, atol=1e-5), f"CPT for {var} does not sum to 1.0")

    def test_04_inference_full_evidence(self):
        """Verify probabilistic inference with full patient profile."""
        engine = BayesianDiagnosisInference(
            str(PROJECT_ROOT / "models" / "bayesian_network.pkl"),
            str(PROJECT_ROOT / "models" / "preprocessor.json")
        )

        case = {
            "age": 60,
            "sex": "Male",
            "cp": "asymptomatic",
            "trestbps": 150,
            "chol": 260,
            "fbs": True,
            "restecg": "lv hypertrophy",
            "thalch": 120,
            "exang": True,
            "oldpeak": 2.0,
            "slope": "flat",
            "ca": 2,
            "thal": "reversable defect"
        }

        res = engine.predict_probability(case)
        self.assertIn("probability_disease", res)
        self.assertTrue(0.0 <= res["probability_disease"] <= 1.0)
        self.assertTrue(0.0 <= res["probability_no_disease"] <= 1.0)
        self.assertTrue(np.isclose(res["probability_disease"] + res["probability_no_disease"], 1.0, atol=1e-4))
        self.assertIn(res["predicted_class"], [0, 1])
        self.assertIn(res["risk_level"], ["Low", "Mild / Borderline", "Moderate to High", "High / Critical"])

    def test_05_inference_partial_evidence_marginalization(self):
        """Verify that Bayesian network correctly marginalizes unobserved variables."""
        engine = BayesianDiagnosisInference(
            str(PROJECT_ROOT / "models" / "bayesian_network.pkl"),
            str(PROJECT_ROOT / "models" / "preprocessor.json")
        )

        # Only 2 variables observed
        partial_case = {
            "age": 30,
            "sex": "Female"
        }

        res = engine.predict_probability(partial_case)
        self.assertEqual(res["evidence_count"], 2)
        self.assertEqual(len(res["unobserved_variables"]), 11)
        self.assertTrue(0.0 <= res["probability_disease"] <= 1.0)

    def test_06_fastapi_endpoints(self):
        """Test FastAPI endpoints via TestClient."""
        with TestClient(app) as client:
            # 1. Health endpoint
            res_health = client.get("/health")
            self.assertEqual(res_health.status_code, 200)
            data_h = res_health.json()
            self.assertEqual(data_h["status"], "healthy")
            self.assertTrue(data_h["model_loaded"])

            # 2. Metadata endpoint
            res_meta = client.get("/metadata")
            self.assertEqual(res_meta.status_code, 200)
            data_m = res_meta.json()
            self.assertIn("nodes", data_m)
            self.assertIn("test_metrics", data_m)

            # 3. Predict endpoint
            payload = {
                "age": 62,
                "sex": "Male",
                "cp": "asymptomatic",
                "trestbps": 140,
                "chol": 240,
                "thalch": 130,
                "exang": True,
                "oldpeak": 2.2
            }
            res_pred = client.post("/predict", json=payload)
            self.assertEqual(res_pred.status_code, 200)
            pred_data = res_pred.json()
            self.assertIn("probability_disease", pred_data)
            self.assertGreater(pred_data["probability_disease"], 0.5)
            self.assertIn("latency_ms", pred_data)

    def test_07_fastapi_validation_error(self):
        """Verify that invalid categorical inputs trigger Pydantic validation error."""
        with TestClient(app) as client:
            invalid_payload = {
                "sex": "Alien",  # Invalid sex
                "age": 45
            }
            res = client.post("/predict", json=invalid_payload)
            self.assertEqual(res.status_code, 422)


if __name__ == "__main__":
    unittest.main()
