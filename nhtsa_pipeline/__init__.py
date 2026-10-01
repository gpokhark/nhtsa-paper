"""
NHTSA Complaint Defect Detection Pipeline
==========================================

LLM-powered extraction, temporal clustering, and recall gap analysis
from NHTSA consumer complaint data.

Usage:
    # Download complaint data
    python -m nhtsa_pipeline.data.download --out data/

    # Extract defect annotations
    python -m nhtsa_pipeline.extract \
        --input data/COMPLAINTS_RECEIVED_2020-2024.zip \
        --output data/annotations.parquet \
        --batch-size 10000

    # Cluster defect signals
    python -m nhtsa_pipeline.cluster \
        --input data/annotations.parquet \
        --output data/clusters.parquet \
        --min-size 50

    # Analyze recall gaps
    python -m nhtsa_pipeline.gap_analysis \
        --clusters data/clusters.parquet \
        --recalls data/recalls.csv \
        --output data/gaps.json

"""

from nhtsa_pipeline.__version__ import __version__

__all__ = ["__version__"]
