# ============================================================
# DeepReads — Combined backend + frontend for HuggingFace Spaces
# Single Space: Next.js on port 7860, FastAPI on port 8001 (internal)
# Next.js proxies /api/* → FastAPI via server-side rewrites
# ============================================================

FROM python:3.11-slim

# System dependencies (Python build tools + Node.js)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Node.js dependencies + build ---
COPY frontend/package.json frontend/package-lock.json* frontend/
RUN cd frontend && npm ci

COPY frontend/ frontend/

# Build Next.js (no NEXT_PUBLIC_API_URL → browser uses /api/* relative path)
# INTERNAL_API_URL tells Next.js server rewrites where FastAPI lives
ENV INTERNAL_API_URL=http://localhost:8001
RUN cd frontend && npm run build

# --- Python source ---
COPY src/ src/
COPY scripts/ scripts/

# Create required directories
RUN mkdir -p models data/processed data/outputs/figures

# --- Startup script ---
COPY start.sh .
RUN chmod +x start.sh

# HF Spaces injects these secrets at runtime:
#   HF_REPO_ID  — artifact repo, e.g. "yashb18/deepreads-artifacts"
#   HF_TOKEN    — optional, if repo is private

EXPOSE 7860
CMD ["./start.sh"]
