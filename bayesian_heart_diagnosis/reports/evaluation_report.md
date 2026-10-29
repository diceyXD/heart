# 📊 Bayesian Network Model Evaluation Report
**Project:** Bayesian Network for Medical Diagnosis (UCI Heart Disease)  
**Model Architecture:** Discrete Bayesian Network with Naive Bayes Topology & Exact Variable Elimination  
**Evaluation Dataset:** Held-out Stratified Test Split (N = 184)  
**Report Generated:** Automatically by `evaluate.py`

---

## 1. Executive Summary & Diagnostic Performance

| Diagnostic Metric | Value | Target Standard | Clinical Interpretation |
|---|---|---|---|
| **Accuracy** | **0.8207** (82.1%) | > 75.0% | Overall correct classification rate |
| **ROC-AUC** | **0.8914** | > 0.800 | Excellent discriminative capacity |
| **Sensitivity (Recall)** | **0.8529** (85.3%) | > 80.0% | Ability to correctly detect true heart disease patients |
| **Specificity** | **0.7805** (78.0%) | > 75.0% | Ability to rule out healthy individuals |
| **Precision (PPV)** | **0.8286** (82.9%) | > 75.0% | Positive predictive value |
| **F1-Score** | **0.8406** | > 0.800 | Harmonic mean of precision and recall |
| **Brier Score** | **0.1433** | < 0.150 | Lower is better; measures probability calibration |
| **Log-Loss** | **0.4792** | < 0.500 | Information-theoretic cross-entropy loss |

---

## 2. Confusion Matrix Breakdown

- **True Negatives (TN):** 64 (Correctly identified without disease)
- **False Positives (FP):** 18 (Healthy patients flagged with risk — potential false alarms)
- **False Negatives (FN):** 15 (Patients with disease missed — critical safety metric)
- **True Positives (TP):** 87 (Correctly diagnosed heart disease)

```
                       Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)             64                       18
Actual Positive (1)             15                       87
```

---

## 3. Probability Calibration Analysis

In medical AI applications, raw binary classification is inadequate: clinicians require trustworthy probability estimates:
- **Observed Brier Score:** `0.1433` indicates well-calibrated posterior probabilities.
- The reliability diagram confirms that when the network predicts an 80% probability of coronary disease, approximately 80% of patients in that risk bin actually possess the condition.
- The model exhibits good calibration across both moderate-risk (40%–60%) and high-risk (>80%) tiers.

---

## 4. Subgroup Fairness & Demographic Audit

### By Sex:
- **Male:** Accuracy = 0.8121 (N = 149)
- **Female:** Accuracy = 0.8571 (N = 35)

### By Age Tier:
- **middle:** Accuracy = 0.8211 (N = 95)
- **senior:** Accuracy = 0.8125 (N = 48)
- **young:** Accuracy = 0.8293 (N = 41)

---

## 5. Artifacts and Generated Visualizations

1. **Confusion Matrix:** `reports/confusion_matrix.png`
2. **Probability Calibration Curve:** `reports/calibration_curve.png`
3. **ROC Curve:** `reports/roc_curve.png`

---

## 6. Safety, Ethics, and Limitations Disclaimer

> **IMPORTANT MEDICAL NOTICE:**  
> This Bayesian Network system is an educational prototype and research demonstrator developed for coursework in Machine Learning with MLOps. It is **NOT** a certified Medical Device and must **NOT** be used as a standalone diagnostic system for patient triage or clinical treatment decisions. All predictions must be evaluated by licensed healthcare professionals.
