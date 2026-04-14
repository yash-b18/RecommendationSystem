# ============================================================
# DeepReads — FastAPI backend for HuggingFace Spaces (Docker)
# Port: 7860 (HF Spaces requirement)
# ============================================================

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/

# Create required directories
RUN mkdir -p models data/processed data/outputs/figures

# HuggingFace Spaces injects these at runtime via Space secrets:
#   HF_REPO_ID  — your model artifact repo, e.g. "yourname/deepreads-artifacts"
#   HF_TOKEN    — (optional) if the artifact repo is private
#   FRONTEND_URL — your Vercel frontend URL (for CORS), e.g. "https://deepreads.vercel.app"

# Download artifacts and start the API server
# The download script is a no-op if files already exist (fast restarts)
CMD python scripts/download_artifacts.py && \
    uvicorn src.api.app:app \
        --host 0.0.0.0 \
        --port 7860 \
        --workers 1 \
        --timeout-keep-alive 75
