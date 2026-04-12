# ============================================================
# Makefile — Explainable Multi-Stage Recommendation System
# All commands assume the venv is activated:
#   source venv/bin/activate
# ============================================================

.PHONY: help setup venv install pipeline download features \
        train-baseline train-classical train-deep \
        evaluate experiment error-analysis api frontend clean

PYTHON = python
VENV   = venv
PIP    = $(VENV)/bin/pip

help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  venv            Create virtual environment"
	@echo "  install         Install all dependencies into venv"
	@echo "  setup           venv + install (full bootstrap)"
	@echo ""
	@echo "Pipeline:"
	@echo "  download        Download & preprocess dataset"
	@echo "  features        Build feature matrix"
	@echo "  train-baseline  Train naive baseline"
	@echo "  train-classical Train LightGBM model"
	@echo "  train-deep      Train Two-Tower model"
	@echo "  evaluate        Evaluate all models"
	@echo "  experiment      Run feature ablation experiment"
	@echo "  error-analysis  Export error analysis"
	@echo "  pipeline        Run full pipeline"
	@echo ""
	@echo "Serving:"
	@echo "  api             Start FastAPI backend"
	@echo "  frontend        Start Next.js frontend dev server"
	@echo ""
	@echo "Utilities:"
	@echo "  clean           Remove generated artifacts"
	@echo ""

venv:
	python3 -m venv $(VENV)
	@echo "Venv created. Run: source venv/bin/activate"

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup: install

download:
	$(PYTHON) scripts/make_dataset.py

features:
	$(PYTHON) scripts/build_features.py

train-baseline:
	$(PYTHON) scripts/train_baseline.py

train-classical:
	$(PYTHON) scripts/train_classical.py

train-deep:
	$(PYTHON) scripts/train_deep.py

evaluate:
	$(PYTHON) scripts/evaluate.py

experiment:
	$(PYTHON) scripts/run_experiment.py

error-analysis:
	$(PYTHON) scripts/error_analysis.py

pipeline:
	$(PYTHON) main.py --all

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

clean:
	rm -rf data/processed/* data/outputs/* models/*.pkl models/*.pt models/*.npy
	@echo "Cleaned processed data, outputs, and model artifacts."
