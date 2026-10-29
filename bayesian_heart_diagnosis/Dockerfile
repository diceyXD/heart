# Multi-stage Dockerfile for Bayesian Network Medical Diagnosis System
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications first for Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY config.yaml .
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY reports/ ./reports/
COPY tests/ ./tests/

# Create logs directory
RUN mkdir -p logs mlruns

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command: run FastAPI backend (can be overridden in docker-compose)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
