"""
streamlit_app.py
----------------
Interactive clinician & educational dashboard for Bayesian Network Heart Disease Diagnosis.

Features:
- Prominent medical safety disclaimer.
- Form inputs for all 13 clinical variables with smart defaults and optional exclusion.
- Dynamic probability meter and color-coded risk stratification.
- Partial evidence visualization: shows which nodes are observed vs marginalized.
- Model transparency tabs:
  1. Diagnostic Inference
  2. Bayesian Network Topology & CPT Inspection
  3. Evaluation Report & Clinical Calibration
  4. MLOps Provenance & Audit Logs
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import pandas as pd
import requests

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import BayesianDiagnosisInference

st.set_page_config(
    page_title="Bayesian Heart Diagnosis System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-high {
        color: #d9534f;
        font-weight: bold;
    }
    .risk-low {
        color: #5cb85c;
        font-weight: bold;
    }
    .disclaimer-banner {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_inference_engine():
    """Cache inference engine instance for high performance."""
    model_path = str(PROJECT_ROOT / "models" / "bayesian_network.pkl")
    prep_path = str(PROJECT_ROOT / "models" / "preprocessor.json")
    return BayesianDiagnosisInference(model_path, prep_path)


def load_metadata() -> Dict[str, Any]:
    meta_path = PROJECT_ROOT / "models" / "model_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_cpt_tables() -> Dict[str, Any]:
    cpt_path = PROJECT_ROOT / "models" / "cpt_tables.json"
    if cpt_path.exists():
        with open(cpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Sidebar & Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=64)
    st.title("Bayesian Heart Diagnostic System")
    st.caption("Probabilistic MLOps Prototype · UCI Multi-Center Dataset")

    nav_tab = st.radio(
        "Navigation",
        [
            "🔬 Diagnostic Inference",
            "🕸️ Network Architecture & CPTs",
            "📊 Evaluation & Calibration",
            "🛡️ MLOps Tracking & Drift"
        ]
    )

    st.markdown("---")
    st.markdown("### System Metadata")
    metadata = load_metadata()
    if metadata:
        st.write(f"**Version:** `{metadata.get('version', '1.0.0')}`")
        st.write(f"**Dataset MD5:** `{metadata.get('data_provenance_md5', '')[:10]}...`")
        st.write(f"**Git Commit:** `{metadata.get('git_commit', '')[:8]}`")
        if "test_evaluation_metrics" in metadata:
            st.write(f"**Test AUC:** `{metadata['test_evaluation_metrics'].get('roc_auc', 0):.4f}`")
            st.write(f"**Test Accuracy:** `{metadata['test_evaluation_metrics'].get('accuracy', 0)*100:.1f}%`")
    st.markdown("---")
    st.caption("Developed for Data Science Audit Course: Machine Learning with MLOps")


# ---------------------------------------------------------------------------
# TAB 1: Diagnostic Inference
# ---------------------------------------------------------------------------
if nav_tab == "🔬 Diagnostic Inference":
    st.title("🫀 Probabilistic Coronary Heart Disease Assessment")

    # Mandatory Clinical Disclaimer Banner
    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <strong>REGULATORY & CLINICAL DISCLAIMER:</strong>
            This application is an educational prototype and research tool constructed as part of an MLOps engineering course.
            It is <strong>NOT</strong> an FDA-approved or CE-marked medical device. All outputs represent statistical probabilities
            derived from historical observational data and must <strong>NEVER</strong> replace clinical judgment, physician consultation,
            or emergency medical care.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "Enter patient evidence below. Unlike rigid black-box models, this **Bayesian Network** "
        "can perform probabilistic inference with **partial evidence** — if a test was not performed, "
        "simply select *(Not Observed)* to marginalize it out."
    )

    col_form, col_results = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.subheader("📋 Patient Diagnostic Profile")

        with st.expander("1. Demographics & Risk Factors", expanded=True):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                include_age = st.checkbox("Include Age", value=True)
                age_val = st.slider("Age (years)", min_value=20, max_value=90, value=55, disabled=not include_age)
            with col_a2:
                include_sex = st.checkbox("Include Sex", value=True)
                sex_val = st.selectbox("Biological Sex", ["Male", "Female"], index=0, disabled=not include_sex)

        with st.expander("2. Symptoms & Physical Vitals", expanded=True):
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                include_cp = st.checkbox("Include Chest Pain", value=True)
                cp_val = st.selectbox(
                    "Chest Pain Type (cp)",
                    ["asymptomatic", "typical angina", "atypical angina", "non-anginal"],
                    index=0,
                    disabled=not include_cp,
                    help="Asymptomatic CAD is clinically the strongest predictor of severe ischemia."
                )
                include_trestbps = st.checkbox("Include Resting Blood Pressure", value=True)
                trestbps_val = st.slider("Resting BP (trestbps) mmHg", 80, 200, 135, disabled=not include_trestbps)
            with col_b2:
                include_exang = st.checkbox("Include Exercise Angina", value=True)
                exang_val = st.selectbox(
                    "Exercise Induced Angina (exang)",
                    ["True (Present)", "False (Absent)"],
                    index=0,
                    disabled=not include_exang
                )
                include_chol = st.checkbox("Include Serum Cholesterol", value=True)
                chol_val = st.slider("Cholesterol (mg/dl)", 100, 500, 240, disabled=not include_chol)

        with st.expander("3. Electrocardiogram & Stress Test", expanded=True):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                include_thalch = st.checkbox("Include Max Heart Rate", value=True)
                thalch_val = st.slider("Max HR Achieved (thalch)", 60, 220, 130, disabled=not include_thalch)
                include_oldpeak = st.checkbox("Include ST Depression", value=True)
                oldpeak_val = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.5, step=0.1, disabled=not include_oldpeak)
            with col_c2:
                include_slope = st.checkbox("Include ST Slope", value=True)
                slope_val = st.selectbox("Peak Exercise ST Slope", ["flat", "upsloping", "downsloping"], index=0, disabled=not include_slope)
                include_restecg = st.checkbox("Include Resting ECG", value=False)
                restecg_val = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"], index=0, disabled=not include_restecg)

        with st.expander("4. Advanced Cardiac Diagnostics (Invasive / Imaging)", expanded=False):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                include_ca = st.checkbox("Include Fluoroscopy Vessels (ca)", value=False)
                ca_val = st.selectbox("Major Vessels Colored (0-3)", ["0.0", "1.0", "2.0", "3.0"], index=0, disabled=not include_ca)
                include_fbs = st.checkbox("Include Fasting Blood Sugar", value=False)
                fbs_val = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"], index=0, disabled=not include_fbs)
            with col_d2:
                include_thal = st.checkbox("Include Thallium Stress Scan", value=False)
                thal_val = st.selectbox("Thallium Perfusion (thal)", ["normal", "reversable defect", "fixed defect"], index=0, disabled=not include_thal)

        calculate_btn = st.button("🔍 Compute Probabilistic Diagnosis", type="primary", use_container_width=True)

    with col_results:
        st.subheader("📈 Diagnostic Inference Result")

        # Prepare evidence dictionary
        evidence = {}
        if include_age:
            evidence["age"] = age_val
        if include_sex:
            evidence["sex"] = sex_val
        if include_cp:
            evidence["cp"] = cp_val
        if include_trestbps:
            evidence["trestbps"] = trestbps_val
        if include_chol:
            evidence["chol"] = chol_val
        if include_exang:
            evidence["exang"] = True if "True" in exang_val else False
        if include_thalch:
            evidence["thalch"] = thalch_val
        if include_oldpeak:
            evidence["oldpeak"] = oldpeak_val
        if include_slope:
            evidence["slope"] = slope_val
        if include_restecg:
            evidence["restecg"] = restecg_val
        if include_ca:
            evidence["ca"] = ca_val
        if include_fbs:
            evidence["fbs"] = True if fbs_val == "True" else False
        if include_thal:
            evidence["thal"] = thal_val

        engine = get_inference_engine()
        result = engine.predict_probability(evidence)

        # Risk Score Card
        prob_pct = result["percentage_risk"]
        risk_lvl = result["risk_level"]
        risk_color = result["risk_color"]

        st.metric(
            label="Estimated Probability of Heart Disease P(HD = 1 | Evidence)",
            value=f"{prob_pct:.2f}%",
            delta=f"{risk_lvl} Risk Tier",
            delta_color="inverse" if prob_pct > 50 else "normal"
        )

        st.progress(float(result["probability_disease"]))

        # Status Banner
        if result["probability_disease"] >= 0.5:
            st.error(f"🔴 **Diagnostic Prediction:** {result['predicted_label']} ({result['risk_level']})")
        else:
            st.success(f"🟢 **Diagnostic Prediction:** {result['predicted_label']} ({result['risk_level']})")

        st.info(f"💡 **Clinical Guidance:** {result['clinical_guidance']}")

        # Probability breakdown
        col_p0, col_p1 = st.columns(2)
        with col_p0:
            st.metric("P(No Disease | E)", f"{result['probability_no_disease']*100:.2f}%")
        with col_p1:
            st.metric("P(Disease Present | E)", f"{result['probability_disease']*100:.2f}%")

        # Evidence transparency
        st.markdown("---")
        st.write(f"**Evidence Variables Observed ({result['evidence_count']}/13):**")
        st.json(result["evidence_used"])

        if result["unobserved_variables"]:
            st.caption(f"**Unobserved Variables Marginalized Out ({len(result['unobserved_variables'])}):**")
            st.write(", ".join([f"`{u}`" for u in result["unobserved_variables"]]))


# ---------------------------------------------------------------------------
# TAB 2: Bayesian Network Architecture & CPTs
# ---------------------------------------------------------------------------
elif nav_tab == "🕸️ Network Architecture & CPTs":
    st.title("🕸️ Bayesian Network Architecture & Learned CPTs")

    st.write(
        "A **Bayesian Network** represents conditional dependencies between random variables "
        "as a Directed Acyclic Graph (DAG) paired with Conditional Probability Tables (CPTs)."
    )

    col_dag, col_props = st.columns([1.1, 0.9])

    with col_dag:
        st.subheader("📐 Directed Acyclic Graph (DAG) Topology")
        st.markdown("""
        ```mermaid
        flowchart TD
            HD["❤️ heart_disease (Target Condition)"]
            
            HD --> A["age_bin"]
            HD --> S["sex"]
            HD --> CP["cp (Chest Pain)"]
            HD --> BP["trestbps_bin (Resting BP)"]
            HD --> CH["chol_bin (Cholesterol)"]
            HD --> FBS["fbs (Fasting Blood Sugar)"]
            HD --> ECG["restecg (Resting ECG)"]
            HD --> HR["thalch_bin (Max Heart Rate)"]
            HD --> EX["exang (Exercise Angina)"]
            HD --> OP["oldpeak_bin (ST Depression)"]
            HD --> SL["slope (Peak ST Slope)"]
            HD --> CA["ca (Fluoroscopy Vessels)"]
            HD --> TH["thal (Thallium Perfusion)"]

            classDef target fill:#d9534f,stroke:#333,stroke-width:2px,color:#fff;
            classDef feature fill:#e1ecf4,stroke:#2b5c8f,stroke-width:1.5px,color:#1b365d;
            class HD target;
            class A,S,CP,BP,CH,FBS,ECG,HR,EX,OP,SL,CA,TH feature;
        ```
        """)

    with col_props:
        st.subheader("📋 Structural Rationale & Assumptions")
        st.markdown("""
        1. **Diagnostic Topology:**
           The target condition `heart_disease` serves as the causal root node influencing
           symptom manifestations, physiological vitals, and diagnostic test responses.
        2. **Conditional Independence:**
           Given known heart disease status ($D \in \{0, 1\}$), clinical findings are
           assumed conditionally independent:
           $$P(X_1, \\dots, X_n \\mid D) = \\prod_{i=1}^n P(X_i \\mid D)$$
        3. **Scalability & Tractability:**
           Avoids exponential state explosion in the target CPT, allowing reliable estimation
           from 920 patients while retaining high discriminative accuracy (AUC = 0.89).
        """)

    st.markdown("---")
    st.subheader("📑 Interactive CPT Viewer")
    cpts = load_cpt_tables()
    selected_var = st.selectbox("Select Node to Inspect Learned CPT:", list(cpts.keys()))

    if selected_var in cpts:
        cpt_info = cpts[selected_var]
        st.write(f"**Variable:** `{cpt_info['variable']}` | **Cardinality:** {cpt_info['variable_cardinality']}")
        st.write(f"**Evidence Parents:** `{cpt_info['evidence_nodes'] or 'None (Root Node)'}`")

        # Format probability table
        states = cpt_info["state_names"][selected_var]
        probs = cpt_info["probabilities"]

        if not cpt_info["evidence_nodes"]:
            df_cpt = pd.DataFrame({"State": states, "Prior Probability P(X)": probs})
            st.dataframe(df_cpt, use_container_width=True)
        else:
            parent = cpt_info["evidence_nodes"][0]
            parent_states = cpt_info["state_names"][parent]
            df_cpt = pd.DataFrame(probs, index=states, columns=[f"P(X | {parent}={ps})" for ps in parent_states])
            st.dataframe(df_cpt, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: Evaluation & Calibration
# ---------------------------------------------------------------------------
elif nav_tab == "📊 Evaluation & Calibration":
    st.title("📊 Model Performance, Validation & Calibration")

    meta = load_metadata()
    test_metrics = meta.get("test_evaluation_metrics", {})

    if test_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.4f}")
        c2.metric("Accuracy", f"{test_metrics.get('accuracy', 0)*100:.2f}%")
        c3.metric("Sensitivity (Recall)", f"{test_metrics.get('sensitivity', 0)*100:.2f}%")
        c4.metric("Brier Score", f"{test_metrics.get('brier_score', 0):.4f}")

    col_p1, col_p2, col_p3 = st.columns(3)

    cm_img = PROJECT_ROOT / "reports" / "confusion_matrix.png"
    cal_img = PROJECT_ROOT / "reports" / "calibration_curve.png"
    roc_img = PROJECT_ROOT / "reports" / "roc_curve.png"

    with col_p1:
        st.subheader("Confusion Matrix")
        if cm_img.exists():
            st.image(str(cm_img), use_container_width=True)

    with col_p2:
        st.subheader("Probability Calibration Curve")
        if cal_img.exists():
            st.image(str(cal_img), use_container_width=True)

    with col_p3:
        st.subheader("ROC Curve")
        if roc_img.exists():
            st.image(str(roc_img), use_container_width=True)

    st.markdown("---")
    st.subheader("📄 Full Markdown Evaluation Report")
    eval_rep = PROJECT_ROOT / "reports" / "evaluation_report.md"
    if eval_rep.exists():
        with open(eval_rep, "r", encoding="utf-8") as f:
            st.markdown(f.read())


# ---------------------------------------------------------------------------
# TAB 4: MLOps Tracking & Drift
# ---------------------------------------------------------------------------
elif nav_tab == "🛡️ MLOps Tracking & Drift":
    st.title("🛡️ MLOps Lifecycle, Audit Logs & Drift Monitoring")

    st.markdown("""
    ### 1. Model Lifecycle & Provenance Architecture
    This project incorporates all key stages of reproducible MLOps deployment:
    - **Dataset Provenance:** Cryptographic MD5 checksum verification of the UCI raw data.
    - **Experiment Tracking:** MLflow tracking backend storing hyper-parameters, metrics, and serialized artifacts.
    - **Versioned Artifacts:** Dual versioning of network graph (`bayesian_network.pkl`) and interpretable CPT tables (`cpt_tables.json`).
    - **Dockerized Multi-Service Deployment:** Dual containers for FastAPI inference backend and Streamlit UI.
    """)

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.subheader("📦 Model Metadata Snapshot")
        st.json(load_metadata())

    with col_meta2:
        st.subheader("📡 Operational Prediction Audit Log (Live)")
        audit_file = PROJECT_ROOT / "logs" / "predictions.jsonl"
        if audit_file.exists():
            lines = []
            with open(audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lines.append(json.loads(line.strip()))
            if lines:
                df_audit = pd.DataFrame(lines).tail(20)
                st.dataframe(df_audit, use_container_width=True)
                st.caption(f"Displaying latest {len(df_audit)} prediction events.")
            else:
                st.info("No prediction events logged yet.")
        else:
            st.info("Log file `logs/predictions.jsonl` will be populated as queries arrive.")

    st.markdown("---")
    st.subheader("🔄 Data Drift & Model Retraining Strategy")
    st.markdown("""
    | Monitoring Dimension | Detection Mechanism | Trigger Threshold | Recommended Action |
    |---|---|---|---|
    | **Input Data Drift** | Kolmogorov-Smirnov test (continuous) & Chi-Square test (categorical) vs `data_snapshot.csv` | p-value < 0.05 on 100+ new cases | Re-assess discretization cutoffs & recalibrate |
    | **Target Concept Drift** | Rolling Brier Score & Log-Loss on audited test labels | Brier score exceeds 0.20 | Re-fit CPTs using Bayesian parameter updating |
    | **API Service Health** | P95 Latency & 5xx HTTP Error Rate from `logs/api.log` | Latency > 150ms or Error rate > 1% | Auto-restart uvicorn workers or scale containers |
    """)
