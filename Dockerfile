# Multi-stage Dockerfile for Bayesian Network Medical Diagnosis System
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files from bayesian_heart_diagnosis/
COPY bayesian_heart_diagnosis/config.yaml .
COPY bayesian_heart_diagnosis/src/ ./src/
COPY bayesian_heart_diagnosis/app/ ./app/
COPY bayesian_heart_diagnosis/models/ ./models/
COPY bayesian_heart_diagnosis/data/ ./data/
COPY bayesian_heart_diagnosis/reports/ ./reports/
COPY bayesian_heart_diagnosis/tests/ ./tests/
COPY streamlit_app.py .

RUN mkdir -p logs mlruns

EXPOSE 8000 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
