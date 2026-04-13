# DeepReads — Explainable Book Recommendations

> Amazon Reviews 2023 · Books · Two-Tower neural model · Metadata-grounded explanations · Next.js demo app

---

## Overview

DeepReads is a production-quality recommendation system built on the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) Books dataset. It implements and compares three modeling tiers, with the Two-Tower neural model serving as the primary recommender in the live app.

| Model | Type | Role |
|---|---|---|
| Popularity Baseline | Naive | Candidate retrieval / baseline |
| LightGBM | Classical ML | Feature-based reranker |
| Two-Tower NN | Deep Learning | Primary recommender (BPR loss) |

**Dataset stats:** 100K+ books · 80K+ users · 5M+ ratings

### Key Design Decisions
- **Two-Tower architecture** with BPR (Bayesian Personalized Ranking) loss for learning user/item embeddings
- **Metadata-grounded language explanations** for the neural model — author match, title keyword overlap, rating tier, and price affinity derived from item metadata (no LLM)
- **SHAP explanations** for the LightGBM model — feature-level attribution
- **Untitled item filtering** — 42% of catalog items have no title; all are excluded from every recommendation path at inference time
- **Cold-start handling** — users without history can pick books manually; their mean item embedding serves as a proxy user vector

---

## Dataset

Data is downloaded automatically from HuggingFace during the pipeline. **No manual download required.**

```bash
python main.py --download
```

This streams the Books review and metadata splits, filters, deduplicates, and saves to `data/raw/` and `data/processed/`.

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
pip install -e .

# 4. Copy environment file
cp .env.example .env
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

After training, outputs are written to `data/outputs/`:

| File | Contents |
|---|---|
| `metrics.csv` | Recall@10, NDCG@10, MRR, Hit Rate for all three models |
| `metrics_table.md` | Markdown table for the report |
| `figures/` | Comparison plots, feature importance, ablation curves |
| `experiment/ablation_results.csv` | Feature ablation results |
| `error_analysis.csv` | Misprediction examples |

**Evaluation protocol:** Leave-last-out — each user's most recent interaction is held out as the test item.

---

## Running the Application

### Backend (FastAPI)
```bash
source venv/bin/activate
uvicorn src.api.app:app --host 0.0.0.0 --port 8001 --reload
```

API docs: `http://localhost:8001/docs`

### Frontend (Next.js)
```bash
cd frontend
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL=http://localhost:8001
npm run dev
```

App: `http://localhost:3000`

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
│   ├── explainability/            ← SHAP (classical) + language (deep) explanations
│   ├── api/                       ← FastAPI app + inference orchestrator
│   └── utils/                     ← logging, seeding, helpers
├── models/                        ← trained model artifacts
├── data/
│   ├── raw/                       ← downloaded data
│   ├── processed/                 ← features & splits
│   └── outputs/                   ← metrics, figures, reports
├── notebooks/                     ← EDA notebooks
├── frontend/                      ← Next.js web app (DeepReads)
└── docs/
    └── report_outline.md          ← report template
```

---

## Deployment

### Backend
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

Set env vars: `API_HOST`, `API_PORT`. Deployable to Railway, Render, or any VM.

### Frontend
```bash
cd frontend && npm run build
# Deploy to Vercel: vercel --prod
```

Set `NEXT_PUBLIC_API_URL` in Vercel to point to your backend.

---

## Limitations
- Cold-start users (no history) receive recommendations via mean-pooled item embeddings from their manually selected books.
- Recall metrics are low due to dataset sparsity (~65 ratings/user on average) and the difficulty of the leave-last-out protocol on a 100K-item catalog.
- Text feature quality is dependent on product title availability (42% of items have no title and are excluded from recommendations).

## Ethics
- **Popularity bias:** mitigated by reranking with the Two-Tower model
- **User privacy:** only encoded integer IDs are used; no real user identifiers stored
- **Transparency:** every recommendation includes a human-readable explanation grounded in observable metadata

---

## Attribution

Developed for Duke University AIPI 540 (Deep Learning Applications), Spring 2025.
AI tools (Claude Code) were used for code generation assistance; all design decisions, model choices, and experimental design are original.
