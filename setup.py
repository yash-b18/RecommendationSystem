"""
Setup script for the recommendation system project.

Usage (inside activated venv):
    pip install -e .

This installs the `src` package in editable mode so all scripts
can import from `src.*` without path manipulation.
"""

from setuptools import find_packages, setup

setup(
    name="recsys",
    version="0.1.0",
    description="Explainable Multi-Stage E-Commerce Recommendation System",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[],   # managed via requirements.txt
)
