# DeepReads — Explainable Book Recommendations

**Two-Tower neural recommender on the Amazon Reviews 2023 Books split, compared head-to-head against a LightGBM reranker and a popularity baseline, with metadata-grounded explanations for every prediction.**

## What it does

Recommends books to users from a 100K-item catalog and returns a plain-English rationale alongside each recommendation. Neural-model rationales are derived from observable item metadata — author match, title-keyword overlap, rating tier, and price affinity — so they reflect the model's actual signal rather than a separate LLM rationalization. The classical model returns SHAP feature attributions per recommendation. Cold-start users without history seed a proxy user vector by mean-pooling the embeddings of books they manually select.

## Data

Amazon Reviews 2023 Books split, streamed from HuggingFace: 100K+ books, 80K+ users, 5M+ ratings, ~65 ratings/user on average. 42% of catalog items have no title and are filtered out at every stage of training, evaluation, and inference. Evaluation uses a leave-last-out protocol — each user's most recent interaction is held out as the test item.

## Models

- **Popularity baseline** — naive global popularity, used for candidate retrieval and as an evaluation floor.
- **LightGBM reranker** — gradient-boosted classifier over engineered user/item/interaction features, with SHAP feature attributions for explanations.
- **Two-Tower neural network** — separate user and item towers trained with Bayesian Personalized Ranking (BPR) loss to learn aligned embedding spaces.

## Training

End-to-end pipeline orchestrated through `main.py`: download → feature build → train (baseline / classical / deep) → evaluate → feature-ablation experiment → error analysis. Reported metrics: Recall@10, NDCG@10, MRR, and Hit Rate, written to `data/outputs/metrics.csv` with comparison plots, feature-importance charts, and ablation curves. A debug mode runs the full pipeline on a small subset in minutes.

## Stack

PyTorch · LightGBM · SHAP · pandas · HuggingFace datasets · FastAPI · Next.js · Docker.
