#!/bin/bash
set -e

# Download model artifacts from HF Hub (skips if already present)
python scripts/download_artifacts.py

# Start FastAPI on internal port 8001
uvicorn src.api.app:app \
    --host 127.0.0.1 \
    --port 8001 \
    --workers 1 \
    --timeout-keep-alive 75 &

# Start Next.js on port 7860 (HF Spaces requirement)
cd frontend
PORT=7860 npm start
