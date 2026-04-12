"""
Main entry point for the recommendation system pipeline.

Orchestrates the full pipeline from data download to evaluation.
Individual stages can also be run directly via scripts/.

Usage (inside activated venv):
    # Full pipeline
    python main.py --all

    # Individual stages
    python main.py --download
    python main.py --features
    python main.py --train-baseline
    python main.py --train-classical
    python main.py --train-deep
    python main.py --evaluate
    python main.py --experiment

    # Debug mode (small subset, fast)
    python main.py --all --debug
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="Recommendation system pipeline orchestrator.")
console = Console()

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _run(script: str, extra_args: list[str] | None = None) -> None:
    """Run a script in the current Python environment."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + (extra_args or [])
    console.print(f"[bold cyan]Running:[/] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        console.print(f"[bold red]Stage failed:[/] {script}")
        raise SystemExit(result.returncode)


@app.command()
def run(
    download: bool = typer.Option(False, "--download", help="Download and preprocess dataset."),
    features: bool = typer.Option(False, "--features", help="Build feature matrix."),
    train_baseline: bool = typer.Option(False, "--train-baseline", help="Train naive baseline."),
    train_classical: bool = typer.Option(False, "--train-classical", help="Train LightGBM model."),
    train_deep: bool = typer.Option(False, "--train-deep", help="Train Two-Tower model."),
    evaluate: bool = typer.Option(False, "--evaluate", help="Evaluate all models."),
    experiment: bool = typer.Option(False, "--experiment", help="Run feature ablation experiment."),
    error_analysis: bool = typer.Option(False, "--error-analysis", help="Export error analysis."),
    all_stages: bool = typer.Option(False, "--all", help="Run full pipeline."),
    debug: bool = typer.Option(False, "--debug", help="Use small data subset (fast)."),
) -> None:
    """Run one or more pipeline stages."""
    extra = ["--debug"] if debug else []

    console.print(Panel.fit(
        "[bold]Explainable Multi-Stage Recommendation System[/]\n"
        "Amazon Reviews 2023 — Video Games",
        border_style="cyan",
    ))

    stages = [
        (download or all_stages, "make_dataset.py", "Data download & preprocessing"),
        (features or all_stages, "build_features.py", "Feature engineering"),
        (train_baseline or all_stages, "train_baseline.py", "Naive baseline training"),
        (train_classical or all_stages, "train_classical.py", "LightGBM training"),
        (train_deep or all_stages, "train_deep.py", "Two-Tower training"),
        (evaluate or all_stages, "evaluate.py", "Evaluation"),
        (experiment or all_stages, "run_experiment.py", "Feature ablation experiment"),
        (error_analysis or all_stages, "error_analysis.py", "Error analysis export"),
    ]

    for should_run, script, label in stages:
        if should_run:
            console.rule(f"[bold green]{label}")
            _run(script, extra)

    console.print(Panel.fit("[bold green]Pipeline complete!", border_style="green"))


if __name__ == "__main__":
    app()
