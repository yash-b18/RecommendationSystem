#!/bin/bash
set -e

# Download model artifacts from HF Hub (skips if already present)
python scripts/download_artifacts.py

# Start FastAPI on internal port 8001 (background)
uvicorn src.api.app:app \
    --host 127.0.0.1 \
    --port 8001 \
    --workers 1 \
    --timeout-keep-alive 75 &

# Wait for FastAPI to finish loading models before starting Next.js
echo "Waiting for FastAPI to be ready..."
until curl -s http://127.0.0.1:8001/health > /dev/null 2>&1; do
    sleep 3
done
echo "FastAPI ready. Starting Next.js..."

# Start Next.js on port 7860 (HF Spaces requirement)
cd frontend
PORT=7860 npm start
