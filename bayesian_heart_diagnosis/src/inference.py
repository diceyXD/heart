"""
inference.py
------------
Inference engine for Bayesian Network Medical Diagnosis.

Features:
- Handles both raw continuous measurements (discretizing on-the-fly) and categorical evidence.
- Supports PARTIAL evidence: any combination of 0 to 13 variables can be provided;
  pgmpy VariableElimination marginalizes out all unobserved variables.
- Returns calibrated probabilities, predicted class, risk classification,
  and active evidence summary.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
from pgmpy.inference import VariableElimination

logger = logging.getLogger(__name__)


class BayesianDiagnosisInference:
    """Thread-safe probabilistic inference service wrapping the trained Bayesian Network."""

    def __init__(
        self,
        model_path: str = "models/bayesian_network.pkl",
        preprocessor_path: str = "models/preprocessor.json"
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self._load_artifacts()

    def _load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor metadata not found at {self.preprocessor_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.preprocessor_path, "r", encoding="utf-8") as f:
            self.prep_meta = json.load(f)

        self.infer = VariableElimination(self.model)
        self.target_col = self.prep_meta.get("target_col", "heart_disease")
        self.valid_nodes = set(self.model.nodes()) - {self.target_col}

        # Cache valid state names for every feature from model CPDs
        self.state_names_map = {}
        for cpd in self.model.get_cpds():
            if cpd.variable != self.target_col:
                self.state_names_map[cpd.variable] = list(cpd.state_names[cpd.variable])

        logger.info(
            f"BayesianDiagnosisInference initialized with {len(self.valid_nodes)} evidence features. "
            f"Target: '{self.target_col}'"
        )

    def discretize_evidence(self, raw_input: Dict[str, Any]) -> Dict[str, str]:
        """
        Map raw patient metrics (numeric and categorical) to discrete network states.
        Omitted or None fields are left unobserved (marginalized during inference).
        """
        evidence = {}
        stats = self.prep_meta.get("train_statistics", {})
        medians = stats.get("medians", {})

        # Age
        if "age" in raw_input and raw_input["age"] is not None:
            age = float(raw_input["age"])
            if age <= 45:
                evidence["age_bin"] = "young"
            elif age <= 60:
                evidence["age_bin"] = "middle"
            else:
                evidence["age_bin"] = "senior"
        elif "age_bin" in raw_input and raw_input["age_bin"]:
            evidence["age_bin"] = str(raw_input["age_bin"])

        # Sex
        if "sex" in raw_input and raw_input["sex"]:
            evidence["sex"] = str(raw_input["sex"])

        # Chest Pain Type
        if "cp" in raw_input and raw_input["cp"]:
            evidence["cp"] = str(raw_input["cp"])

        # Resting Blood Pressure (trestbps)
        if "trestbps" in raw_input and raw_input["trestbps"] is not None:
            bp = float(raw_input["trestbps"])
            if bp <= 0:
                bp = medians.get("trestbps", 130.0)
            if bp <= 120:
                evidence["trestbps_bin"] = "normal"
            elif bp <= 140:
                evidence["trestbps_bin"] = "elevated"
            else:
                evidence["trestbps_bin"] = "high"
        elif "trestbps_bin" in raw_input and raw_input["trestbps_bin"]:
            evidence["trestbps_bin"] = str(raw_input["trestbps_bin"])

        # Serum Cholesterol (chol)
        if "chol" in raw_input and raw_input["chol"] is not None:
            chol = float(raw_input["chol"])
            if chol <= 0:
                chol = medians.get("chol", 223.0)
            if chol <= 200:
                evidence["chol_bin"] = "normal"
            elif chol <= 240:
                evidence["chol_bin"] = "borderline"
            else:
                evidence["chol_bin"] = "high"
        elif "chol_bin" in raw_input and raw_input["chol_bin"]:
            evidence["chol_bin"] = str(raw_input["chol_bin"])

        # Fasting Blood Sugar (fbs)
        if "fbs" in raw_input and raw_input["fbs"] is not None:
            val = raw_input["fbs"]
            evidence["fbs"] = "True" if str(val).lower() in ["true", "1", "yes"] else "False"

        # Resting ECG
        if "restecg" in raw_input and raw_input["restecg"]:
            evidence["restecg"] = str(raw_input["restecg"])

        # Maximum Heart Rate (thalch)
        if "thalch" in raw_input and raw_input["thalch"] is not None:
            hr = float(raw_input["thalch"])
            if hr <= 120:
                evidence["thalch_bin"] = "low"
            elif hr <= 150:
                evidence["thalch_bin"] = "moderate"
            else:
                evidence["thalch_bin"] = "high"
        elif "thalch_bin" in raw_input and raw_input["thalch_bin"]:
            evidence["thalch_bin"] = str(raw_input["thalch_bin"])

        # Exercise Induced Angina (exang)
        if "exang" in raw_input and raw_input["exang"] is not None:
            val = raw_input["exang"]
            evidence["exang"] = "True" if str(val).lower() in ["true", "1", "yes"] else "False"

        # ST Depression (oldpeak)
        if "oldpeak" in raw_input and raw_input["oldpeak"] is not None:
            op = float(raw_input["oldpeak"])
            if op <= 0.0:
                evidence["oldpeak_bin"] = "none"
            elif op <= 1.5:
                evidence["oldpeak_bin"] = "mild"
            else:
                evidence["oldpeak_bin"] = "severe"
        elif "oldpeak_bin" in raw_input and raw_input["oldpeak_bin"]:
            evidence["oldpeak_bin"] = str(raw_input["oldpeak_bin"])

        # ST Slope
        if "slope" in raw_input and raw_input["slope"]:
            evidence["slope"] = str(raw_input["slope"])

        # Major Vessels (ca)
        if "ca" in raw_input and raw_input["ca"] is not None:
            try:
                ca_val = str(float(raw_input["ca"]))
                evidence["ca"] = ca_val
            except (ValueError, TypeError):
                evidence["ca"] = str(raw_input["ca"])

        # Thalassemia (thal)
        if "thal" in raw_input and raw_input["thal"]:
            evidence["thal"] = str(raw_input["thal"])

        # Validate that all discretized evidence states exist in the model's domain
        cleaned_evidence = {}
        for var, state in evidence.items():
            if var in self.state_names_map:
                valid_states = self.state_names_map[var]
                if state in valid_states:
                    cleaned_evidence[var] = state
                else:
                    logger.warning(
                        f"State '{state}' not recognized for variable '{var}'. "
                        f"Valid states: {valid_states}. Omitting this variable from evidence."
                    )

        return cleaned_evidence

    def predict_probability(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute probabilistic inference for the given evidence.
        Marginalizes unobserved variables automatically.
        """
        evidence = self.discretize_evidence(raw_input)

        try:
            q = self.infer.query(
                variables=[self.target_col],
                evidence=evidence,
                show_progress=False
            )
            states = list(q.state_names[self.target_col])
            idx_1 = states.index("1") if "1" in states else 1
            idx_0 = states.index("0") if "0" in states else 0

            prob_disease = float(q.values[idx_1])
            prob_no_disease = float(q.values[idx_0])
        except Exception as e:
            logger.error(f"Inference computation error with evidence {evidence}: {e}")
            raise RuntimeError(f"Probabilistic inference failed: {str(e)}")

        predicted_class = 1 if prob_disease >= 0.5 else 0

        # Clinical risk stratification
        if prob_disease < 0.20:
            risk_level = "Low"
            risk_color = "green"
            clinical_guidance = "Low probability of significant coronary artery disease based on provided evidence."
        elif prob_disease < 0.50:
            risk_level = "Mild / Borderline"
            risk_color = "orange"
            clinical_guidance = "Moderate risk. Standard non-invasive cardiovascular assessment advised."
        elif prob_disease < 0.75:
            risk_level = "Moderate to High"
            risk_color = "darkorange"
            clinical_guidance = "Elevated risk. Cardiology consultation and diagnostic testing indicated."
        else:
            risk_level = "High / Critical"
            risk_color = "red"
            clinical_guidance = "High probability of coronary disease. Prompt clinical evaluation strongly recommended."

        unobserved = sorted(list(self.valid_nodes - set(evidence.keys())))

        return {
            "probability_disease": round(prob_disease, 4),
            "probability_no_disease": round(prob_no_disease, 4),
            "percentage_risk": round(prob_disease * 100, 2),
            "predicted_class": predicted_class,
            "predicted_label": "Heart Disease Present" if predicted_class == 1 else "No Heart Disease",
            "risk_level": risk_level,
            "risk_color": risk_color,
            "clinical_guidance": clinical_guidance,
            "evidence_used": evidence,
            "evidence_count": len(evidence),
            "unobserved_variables": unobserved,
            "disclaimer": (
                "Educational research prototype. Not approved for diagnostic or clinical treatment decisions."
            )
        }


# Quick CLI test
if __name__ == "__main__":
    engine = BayesianDiagnosisInference()

    # Case 1: High risk patient
    case_high = {
        "age": 67,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 160,
        "chol": 286,
        "thalch": 108,
        "exang": True,
        "oldpeak": 2.5,
        "slope": "flat",
        "ca": "2.0",
        "thal": "reversable defect"
    }
    result_high = engine.predict_probability(case_high)
    print("\n--- Test Case: High Risk ---")
    print(json.dumps(result_high, indent=2))

    # Case 2: Low risk patient
    case_low = {
        "age": 35,
        "sex": "Female",
        "cp": "typical angina",
        "trestbps": 115,
        "chol": 180,
        "thalch": 175,
        "exang": False,
        "oldpeak": 0.0
    }
    result_low = engine.predict_probability(case_low)
    print("\n--- Test Case: Low Risk (Partial Evidence) ---")
    print(json.dumps(result_low, indent=2))
