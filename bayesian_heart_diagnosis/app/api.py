"""
api.py
------
FastAPI production inference service for the Bayesian Network Medical Diagnosis application.

Features:
- Pydantic v2 data validation with informative error messages.
- Endpoints:
  - GET  /         : Service welcome & documentation index
  - GET  /health   : Health check, model status, and version metadata
  - GET  /metadata : Model topology, valid feature domains, and training metadata
  - POST /predict  : Probabilistic inference for an individual patient profile
  - POST /predict/batch : Batch inference for multiple profiles
- Structured operational & monitoring logging:
  - Latency, status codes -> logs/api.log
  - Sanitized predictions -> logs/predictions.jsonl (for data drift monitoring)
  - Exceptions -> logs/errors.log
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Include project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import BayesianDiagnosisInference

# Initialize loggers
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# General API Logger
api_logger = logging.getLogger("api")
api_logger.setLevel(logging.INFO)
api_handler = logging.FileHandler(LOGS_DIR / "api.log", encoding="utf-8")
api_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
api_logger.addHandler(api_handler)
api_logger.addHandler(logging.StreamHandler(sys.stdout))

# Error Logger
error_logger = logging.getLogger("errors")
error_logger.setLevel(logging.ERROR)
err_handler = logging.FileHandler(LOGS_DIR / "errors.log", encoding="utf-8")
err_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
error_logger.addHandler(err_handler)

# Prediction Audit Logger (JSONL for MLOps drift monitoring)
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions.jsonl"


def log_prediction_audit(record: Dict[str, Any]):
    """Log non-sensitive prediction events to JSONL for monitoring."""
    try:
        with open(PREDICTIONS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        error_logger.error(f"Failed to append to prediction audit log: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="Bayesian Network Medical Diagnosis API",
    description=(
        "Probabilistic inference service estimating coronary heart disease probability "
        "using a Bayesian Network on the UCI Heart Disease cohort. "
        "Educational and research prototype."
    ),
    version="1.0.0"
)

# Enable CORS for Streamlit and external frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load inference engine
inference_engine: Optional[BayesianDiagnosisInference] = None
startup_time: Optional[float] = None


@app.on_event("startup")
def startup_event():
    global inference_engine, startup_time
    startup_time = time.time()
    try:
        model_path = str(PROJECT_ROOT / "models" / "bayesian_network.pkl")
        prep_path = str(PROJECT_ROOT / "models" / "preprocessor.json")
        inference_engine = BayesianDiagnosisInference(model_path, prep_path)
        api_logger.info("BayesianDiagnosisInference engine initialized successfully.")
    except Exception as e:
        error_logger.critical(f"Failed to initialize inference engine: {e}", exc_info=True)
        raise RuntimeError(f"Startup initialization failed: {e}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware for tracking request latency, status codes, and errors."""
    start_time = time.time()
    response = None
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        api_logger.info(
            f"{request.method} {request.url.path} - Status: {response.status_code} "
            f"- Latency: {process_time:.2f}ms"
        )
        return response
    except Exception as exc:
        process_time = (time.time() - start_time) * 1000
        error_logger.error(
            f"Unhandled exception during {request.method} {request.url.path} "
            f"(Latency: {process_time:.2f}ms): {exc}",
            exc_info=True
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected server error occurred during inference.",
                "details": str(exc)
            }
        )


# ---------------------------------------------------------------------------
# Pydantic Schemas for Validation
# ---------------------------------------------------------------------------
class PatientEvidenceInput(BaseModel):
    """
    Patient diagnostic evidence input schema.
    All fields are OPTIONAL to fully leverage Bayesian Network's capability
    to perform inference under partial observation.
    """
    age: Optional[float] = Field(None, ge=1, le=120, description="Age in years (e.g. 58)")
    sex: Optional[str] = Field(None, description="Biological sex ('Male' or 'Female')")
    cp: Optional[str] = Field(
        None,
        description="Chest pain type: 'typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'"
    )
    trestbps: Optional[float] = Field(None, ge=50, le=280, description="Resting blood pressure in mmHg (e.g. 130)")
    chol: Optional[float] = Field(None, ge=50, le=700, description="Serum cholesterol in mg/dl (e.g. 240)")
    fbs: Optional[Union[bool, str, int]] = Field(
        None, description="Fasting blood sugar > 120 mg/dl (True/False or 1/0)"
    )
    restecg: Optional[str] = Field(
        None, description="Resting ECG: 'normal', 'st-t abnormality', 'lv hypertrophy'"
    )
    thalch: Optional[float] = Field(None, ge=50, le=250, description="Maximum heart rate achieved (e.g. 150)")
    exang: Optional[Union[bool, str, int]] = Field(
        None, description="Exercise-induced angina (True/False or 1/0)"
    )
    oldpeak: Optional[float] = Field(
        None, ge=-5.0, le=10.0, description="ST depression induced by exercise relative to rest (e.g. 1.5)"
    )
    slope: Optional[str] = Field(
        None, description="Slope of peak exercise ST segment: 'upsloping', 'flat', 'downsloping'"
    )
    ca: Optional[Union[float, int, str]] = Field(
        None, description="Number of major vessels (0-3) colored by fluoroscopy"
    )
    thal: Optional[str] = Field(
        None, description="Thallium heart scan result: 'normal', 'fixed defect', 'reversable defect'"
    )

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        if v is not None and v.capitalize() not in ["Male", "Female"]:
            raise ValueError("Sex must be either 'Male' or 'Female'")
        return v.capitalize() if v else None

    @field_validator("cp")
    @classmethod
    def validate_cp(cls, v):
        valid = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
        if v is not None and v.lower() not in valid:
            raise ValueError(f"Chest pain (cp) must be one of: {valid}")
        return v.lower() if v else None

    @field_validator("restecg")
    @classmethod
    def validate_restecg(cls, v):
        valid = ["normal", "st-t abnormality", "lv hypertrophy"]
        if v is not None and v.lower() not in valid:
            raise ValueError(f"Rest ECG must be one of: {valid}")
        return v.lower() if v else None

    @field_validator("slope")
    @classmethod
    def validate_slope(cls, v):
        valid = ["upsloping", "flat", "downsloping"]
        if v is not None and v.lower() not in valid:
            raise ValueError(f"Slope must be one of: {valid}")
        return v.lower() if v else None

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, v):
        valid = ["normal", "fixed defect", "reversable defect"]
        if v is not None and v.lower() not in valid:
            raise ValueError(f"Thal must be one of: {valid}")
        return v.lower() if v else None

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": "Male",
                "cp": "asymptomatic",
                "trestbps": 145,
                "chol": 233,
                "fbs": False,
                "restecg": "lv hypertrophy",
                "thalch": 150,
                "exang": False,
                "oldpeak": 2.3,
                "slope": "downsloping",
                "ca": 0,
                "thal": "fixed defect"
            }
        }
    }


class BatchEvidenceInput(BaseModel):
    patients: List[PatientEvidenceInput]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def read_root():
    """Service landing page and navigation links."""
    return {
        "service": "Bayesian Network Medical Diagnosis API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "model_metadata": "/metadata",
        "disclaimer": "Educational research prototype. Not for clinical treatment or diagnosis."
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint reporting engine status, uptime, and model version."""
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not loaded."
        )

    uptime_sec = time.time() - startup_time if startup_time else 0
    meta_file = PROJECT_ROOT / "models" / "model_meta.json"
    model_version = "unknown"
    git_commit = "unknown"

    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_version = meta.get("version", "1.0.0")
                git_commit = meta.get("git_commit", "unknown")
        except Exception:
            pass

    return {
        "status": "healthy",
        "uptime_seconds": round(uptime_sec, 2),
        "model_loaded": True,
        "model_version": model_version,
        "git_commit": git_commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metadata", tags=["Model"])
def get_metadata():
    """Retrieve Bayesian Network graph topology, feature domains, and dataset provenance."""
    meta_file = PROJECT_ROOT / "models" / "model_meta.json"
    if not meta_file.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found.")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "project": meta.get("project"),
        "version": meta.get("version"),
        "trained_at_utc": meta.get("trained_at_utc"),
        "git_commit": meta.get("git_commit"),
        "mlflow_run_id": meta.get("mlflow_run_id"),
        "data_provenance_md5": meta.get("data_provenance_md5"),
        "training_rows": meta.get("training_rows"),
        "validation_rows": meta.get("validation_rows"),
        "nodes": meta.get("nodes"),
        "edges_count": len(meta.get("edges", [])),
        "test_metrics": meta.get("test_evaluation_metrics", meta.get("train_metrics")),
        "valid_states": inference_engine.state_names_map if inference_engine else {}
    }


@app.post("/predict", tags=["Inference"])
def predict(evidence: PatientEvidenceInput):
    """
    Perform probabilistic Bayesian inference for a single patient evidence payload.
    Supports both complete and partial evidence profiles.
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not ready."
        )

    raw_dict = evidence.model_dump(exclude_none=True)
    t0 = time.time()

    try:
        result = inference_engine.predict_probability(raw_dict)
        latency_ms = (time.time() - t0) * 1000

        # Log prediction event for drift and monitoring audit
        audit_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_features_count": result["evidence_count"],
            "probability_disease": result["probability_disease"],
            "predicted_class": result["predicted_class"],
            "risk_level": result["risk_level"],
            "latency_ms": round(latency_ms, 2)
        }
        log_prediction_audit(audit_record)

        result["latency_ms"] = round(latency_ms, 2)
        return result

    except Exception as e:
        error_logger.error(f"Inference error on input {raw_dict}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference computation error: {str(e)}"
        )


@app.post("/predict/batch", tags=["Inference"])
def predict_batch_endpoint(batch: BatchEvidenceInput):
    """Perform batch probabilistic inference for multiple patient records."""
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not ready."
        )

    t0 = time.time()
    results = []

    for idx, patient in enumerate(batch.patients):
        raw_dict = patient.model_dump(exclude_none=True)
        try:
            res = inference_engine.predict_probability(raw_dict)
            results.append({"patient_index": idx, "success": True, "prediction": res})
        except Exception as e:
            results.append({"patient_index": idx, "success": False, "error": str(e)})

    total_latency_ms = (time.time() - t0) * 1000

    return {
        "total_cases": len(batch.patients),
        "total_latency_ms": round(total_latency_ms, 2),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)
