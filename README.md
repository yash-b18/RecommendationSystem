# Explainable Multi-Stage E-Commerce Recommendation System

> Amazon Reviews 2023 · Video Games · Three-model comparison · SHAP explainability · Next.js demo app

---

## Overview

This project builds a production-quality recommendation system on the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset (Video Games category). It implements and rigorously compares three modeling tiers:

| Model | Type | Role in Pipeline |
|---|---|---|
| Popularity baseline | Naive | Candidate retrieval |
| LightGBM | Classical ML | Feature-based reranker |
| Two-Tower NN | Deep Learning | End-to-end neural scorer |

### Novelty
Prior work on this dataset uses simple collaborative filtering or basic matrix factorization. This project contributes:
- A **multi-stage pipeline** (retrieval → reranking) comparable to production recommenders
- **Explainable recommendations** via SHAP (classical model) and metadata-grounded natural language (deep model)
- A **feature ablation experiment** quantifying the marginal value of each feature group
- A **polished interactive web app** for investor-style demos

---

## Dataset

Data is downloaded automatically from HuggingFace during the pipeline.

**No manual download required.** The `make_dataset.py` script will:
1. Stream the Video Games review and metadata splits from HuggingFace
2. Filter, deduplicate, and save to `data/raw/`
3. Preprocess into `data/processed/`

If you need a HuggingFace token (for gated datasets), set it in `.env`:
```
HF_TOKEN=your_token_here
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/yash-b18/RecommendationSystem.git
cd RecommendationSystem

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                # installs src as editable package

# 4. Copy environment file
cp .env.example .env
# Edit .env if needed
```

---

## Training the Pipeline

Run the full pipeline end-to-end:
```bash
python main.py --all
```

Or run individual stages:
```bash
python main.py --download          # Download & preprocess dataset
python main.py --features          # Build feature matrix
python main.py --train-baseline    # Train naive popularity model
python main.py --train-classical   # Train LightGBM reranker
python main.py --train-deep        # Train Two-Tower neural model
python main.py --evaluate          # Evaluate all models
python main.py --experiment        # Run feature ablation experiment
python main.py --error-analysis    # Export error analysis
```

**Debug mode** (small subset, runs in minutes):
```bash
python main.py --all --debug
```

---

## Evaluation

After training, evaluation outputs are written to `data/outputs/`:

| File | Contents |
|---|---|
| `metrics.csv` | Recall@K, NDCG@K, HitRate@K, MRR for all models |
| `metrics_table.md` | Markdown table for the report |
| `figures/` | Comparison plots, feature importance, ablation curves |
| `experiment/ablation_results.csv` | Feature ablation results |
| `error_analysis.csv` | 5+ misprediction examples |

---

## Running the Application

### Backend (FastAPI)
```bash
source venv/bin/activate
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`

### Frontend (Next.js)
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

App available at `http://localhost:3000`

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── setup.py
├── main.py                        ← pipeline orchestrator
├── .env.example
├── scripts/
│   ├── make_dataset.py            ← download & preprocess
│   ├── build_features.py          ← feature engineering
│   ├── train_baseline.py          ← naive popularity model
│   ├── train_classical.py         ← LightGBM reranker
│   ├── train_deep.py              ← Two-Tower neural model
│   ├── evaluate.py                ← evaluate all models
│   ├── run_experiment.py          ← feature ablation experiment
│   ├── error_analysis.py          ← export mispredictions
│   └── serve_utils.py             ← model loading for API
├── src/
│   ├── config.py                  ← all configuration
│   ├── data/                      ← data loading & splitting
│   ├── features/                  ← feature engineering modules
│   ├── models/                    ← model implementations
│   ├── evaluation/                ← metrics & evaluation
│   ├── explainability/            ← SHAP + language explanations
│   ├── api/                       ← FastAPI app
│   └── utils/                     ← logging, seeding, helpers
├── models/                        ← trained model artifacts
├── data/
│   ├── raw/                       ← downloaded data
│   ├── processed/                 ← features & splits
│   └── outputs/                   ← metrics, figures, reports
│       └── figures/
├── notebooks/                     ← EDA notebooks (not graded)
├── frontend/                      ← Next.js web app
└── docs/
    └── report_outline.md          ← pre-structured report template
```

---

## Sample Screenshots

> _(To be added after app deployment)_

---

## Deployment

### Backend
```bash
# Build and run with uvicorn
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Alternatively, deploy to Railway, Render, or any VM with the following env vars:
- `API_HOST`
- `API_PORT`

### Frontend
```bash
cd frontend
npm run build
# Deploy to Vercel: vercel --prod
```

Set `NEXT_PUBLIC_API_URL` in Vercel environment variables to point to your backend.

---

## Limitations
- Trained on Video Games reviews only; generalizes to other categories with retraining.
- Cold-start users (< 5 ratings) receive popularity-based fallback recommendations.
- Text feature quality depends on product title length.

## Ethics Considerations
- Popularity bias: popular items dominate naive baseline; mitigated by reranking.
- Filter bubbles: addressed by diversity-aware candidate sampling (future work).
- User privacy: no real user identifiers are stored; only encoded IDs.
- See `docs/report_outline.md` Section 13 for full ethics statement.

---

## Attribution

This project was developed for Duke University AIPI 540 (Deep Learning Applications), Spring 2025.
AI tools (Claude Code) were used for code generation assistance; all design decisions, model choices, and experimental design are original.
