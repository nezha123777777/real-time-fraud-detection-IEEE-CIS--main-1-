# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables for safer execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Copy only serving requirements first to cache layers
COPY deployment/requirements_serving.txt .
RUN pip install --no-cache-dir -r requirements_serving.txt

# Create necessary directories
RUN mkdir -p /app/data/processed /app/artifacts/models /app/reports/artifacts

# Copy source code and deployment code
COPY src /app/src
COPY deployment /app/deployment
COPY configs /app/configs
COPY feature_repo /app/feature_repo

# Create necessary directories for data and Feast
RUN mkdir -p /app/data/interim /app/data/registry /app/data/online_store

# Important: Copy only the finalized artefacts and models
COPY artifacts/models/final /app/artifacts/models/final
COPY artifacts/models/artifacts /app/artifacts/models/artifacts
COPY reports /app/reports

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:$PORT/health || exit 1

EXPOSE 8000

# Run FastAPI via Uvicorn
CMD ["uvicorn", "deployment.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
