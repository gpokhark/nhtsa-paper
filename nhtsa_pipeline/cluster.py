"""
Defect signal clustering and temporal analysis.

Implements:
- Semantic embedding of defect annotations
- Hierarchical clustering into defect signal clusters
- Temporal intensity analysis
- Cluster statistics and visualization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DefectCluster:
    """Represents a defect signal cluster."""

    def __init__(
        self,
        cluster_id: int,
        complaint_ids: list[str],
        defect_category: str,
        avg_severity: float,
        avg_confidence: float,
        size: int,
        date_range: tuple[str, str],
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ):
        self.cluster_id = cluster_id
        self.complaint_ids = complaint_ids
        self.defect_category = defect_category
        self.avg_severity = avg_severity
        self.avg_confidence = avg_confidence
        self.size = size
        self.date_range = date_range
        self.embedding = embedding
        self.metadata = metadata or {}

        # Temporal analysis (computed later)
        self.monthly_intensity: dict[str, int] = {}
        self.first_complaint_date: str = ""
        self.last_complaint_date: str = ""
        self.peak_month: str = ""
        self.peak_intensity: int = 0

    @property
    def is_safety_relevant(self) -> bool:
        return self.avg_severity >= 3.0

    @property
    def is_high_severity(self) -> bool:
        return self.avg_severity >= 4.0

    def compute_temporal_intensity(self, df: pd.DataFrame) -> dict[str, int]:
        """Compute monthly complaint intensity for this cluster.

        Args:
            df: Full annotations DataFrame

        Returns:
            Dictionary mapping YYYY-MM to complaint count
        """
        cluster_df = df[df["complaint_id"].isin(self.complaint_ids)]
        if cluster_df.empty:
            return {}

        cluster_df = cluster_df.copy()
        cluster_df["month"] = pd.to_datetime(cluster_df["date_received"]).dt.to_period("M").astype(str)

        intensity = cluster_df["month"].value_counts().to_dict()
        self.monthly_intensity = intensity
        self.first_complaint_date = cluster_df["date_received"].min()
        self.last_complaint_date = cluster_df["date_received"].max()

        if intensity:
            self.peak_month = max(intensity, key=intensity.get)
            self.peak_intensity = intensity[self.peak_month]

        return intensity

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "complaint_ids": self.complaint_ids,
            "defect_category": self.defect_category,
            "avg_severity": round(self.avg_severity, 2),
            "avg_confidence": round(self.avg_confidence, 2),
            "size": self.size,
            "date_range": self.date_range,
            "first_complaint_date": self.first_complaint_date,
            "last_complaint_date": self.last_complaint_date,
            "peak_month": self.peak_month,
            "peak_intensity": self.peak_intensity,
            "monthly_intensity": self.monthly_intensity,
            "is_safety_relevant": self.is_safety_relevant,
            "is_high_severity": self.is_high_severity,
            **self.metadata,
        }


def compute_embeddings(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    column: str = "defect_category_label",
) -> np.ndarray:
    """Compute semantic embeddings for defect annotations.

    Uses a combination of defect category label and symptom patterns
    to create meaningful cluster boundaries.

    Args:
        df: Annotations DataFrame
        model_name: Sentence transformer model name
        column: Column to embed (defect category label)

    Returns:
        Array of embeddings shape (n_samples, embedding_dim)
    """
    logger.info(f"Computing embeddings with model {model_name}...")

    # Load sentence transformer
    model = SentenceTransformer(model_name)

    # Build embedding text: combine category, subsystem, and key symptoms
    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building embeddings"):
        category = row.get("defect_category_label", "unknown")
        subsystem = row.get("subsystem_label", "unknown")
        symptoms = row.get("symptoms", [])
        severity = row.get("severity", 3)

        # Build embedding text
        symptom_str = ", ".join(symptoms[:5]) if symptoms else "none"
        text = f"Defect: {category}, Subsystem: {subsystem}, Severity: {severity}, Symptoms: {symptom_str}"
        texts.append(text)

    # Compute embeddings
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=256)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms

    logger.info(f"Computed {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings


def cluster_defect_signals(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    distance_threshold: float = 0.45,
) -> list[DefectCluster]:
    """Cluster defect annotations into defect signal clusters.

    Uses Agglomerative Clustering with cosine distance (via normalized embeddings).
    Dynamic threshold to ensure minimum cluster size.

    Args:
        df: Annotations DataFrame
        embeddings: Pre-computed embeddings
        min_cluster_size: Minimum complaints per cluster
        distance_threshold: Maximum distance for cluster merging

    Returns:
        List of DefectCluster objects
    """
    logger.info(f"Clustering {len(df)} annotations (min_cluster_size={min_cluster_size})...")

    # Fit agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        distance_threshold=distance_threshold,
        linkage="average",
    )

    # Compute pairwise cosine distance
    distances = pairwise_distances(embeddings, metric="cosine")

    labels = clustering.fit_predict(distances)

    # Build clusters
    clusters = []
    unique_labels = np.unique(labels)

    for label in tqdm(unique_labels, desc="Building clusters"):
        mask = labels == label
        cluster_complaint_ids = df.loc[mask, "complaint_id"].tolist()

        if len(cluster_complaint_ids) < min_cluster_size:
            logger.debug(f"Skipping cluster {label}: size {len(cluster_complaint_ids)} < {min_cluster_size}")
            continue

        cluster_df = df.loc[mask]
        category = cluster_df["defect_category"].mode().iloc[0]
        avg_severity = cluster_df["severity"].mean()
        avg_confidence = cluster_df["confidence"].mean()

        # Date range
        dates = pd.to_datetime(cluster_df["date_received"])
        date_min = dates.min().strftime("%Y-%m-%d")
        date_max = dates.max().strftime("%Y-%m-%d")

        # Cluster centroid embedding
        centroid = embeddings[mask].mean(axis=0)

        cluster = DefectCluster(
            cluster_id=int(label),
            complaint_ids=cluster_complaint_ids,
            defect_category=category,
            avg_severity=avg_severity,
            avg_confidence=avg_confidence,
            size=len(cluster_complaint_ids),
            date_range=(date_min, date_max),
            embedding=centroid,
        )
        clusters.append(cluster)

    # Sort by size (largest first)
    clusters.sort(key=lambda c: c.size, reverse=True)

    # Re-assign cluster IDs
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i

    logger.info(f"Identified {len(clusters)} defect signal clusters")
    logger.info(f"  Total complaints in clusters: {sum(c.size for c in clusters):,}")
    logger.info(f"  Largest cluster: {clusters[0].size:,} complaints" if clusters else "  No clusters")

    return clusters


def analyze_temporal_patterns(
    clusters: list[DefectCluster],
    df: pd.DataFrame,
) -> list[DefectCluster]:
    """Compute temporal intensity for all clusters.

    Args:
        clusters: List of DefectCluster objects
        df: Full annotations DataFrame

    Returns:
        Updated list of DefectCluster objects with temporal analysis
    """
    logger.info(f"Analyzing temporal patterns for {len(clusters)} clusters...")

    for cluster in tqdm(clusters, desc="Temporal analysis"):
        cluster.compute_temporal_intensity(df)

    return clusters


def get_cluster_statistics(clusters: list[DefectCluster]) -> pd.DataFrame:
    """Compute summary statistics for all clusters.

    Args:
        clusters: List of DefectCluster objects

    Returns:
        DataFrame with cluster statistics
    """
    stats = []
    for cluster in clusters:
        stats.append({
            "cluster_id": cluster.cluster_id,
            "size": cluster.size,
            "defect_category": cluster.defect_category,
            "avg_severity": cluster.avg_severity,
            "avg_confidence": cluster.avg_confidence,
            "is_safety_relevant": cluster.is_safety_relevant,
            "is_high_severity": cluster.is_high_severity,
            "first_complaint": cluster.first_complaint_date,
            "last_complaint": cluster.last_complaint_date,
            "peak_month": cluster.peak_month,
            "peak_intensity": cluster.peak_intensity,
            "date_span_months": _months_between(cluster.first_complaint_date, cluster.last_complaint_date),
        })

    return pd.DataFrame(stats)


def _months_between(date1: str, date2: str) -> int:
    """Calculate months between two date strings (YYYY-MM-DD format)."""
    d1 = pd.to_datetime(date1)
    d2 = pd.to_datetime(date2)
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def export_clusters(
    clusters: list[DefectCluster],
    output_path: str | Path,
) -> None:
    """Export cluster data to JSON and Parquet.

    Args:
        clusters: List of DefectCluster objects
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON export (full metadata)
    data = [c.to_dict() for c in clusters]
    import json
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Parquet export (tabular)
    stats = get_cluster_statistics(clusters)
    stats.to_parquet(output_path.with_suffix(".parquet"), index=False)

    logger.info(f"Exported {len(clusters)} clusters to {output_path}")


def visualize_cluster_distribution(
    clusters: list[DefectCluster],
    output_path: str | Path | None = None,
) -> None:
    """Create visualization of cluster distribution.

    Args:
        clusters: List of DefectCluster objects
        output_path: Output image path (None to show)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    stats = get_cluster_statistics(clusters)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cluster size distribution
    axes[0, 0].hist(stats["size"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Cluster Size (complaints)")
    axes[0, 0].set_ylabel("Number of Clusters")
    axes[0, 0].set_title("Cluster Size Distribution")
    axes[0, 0].set_xscale("log")

    # 2. Severity distribution
    severity_counts = stats["avg_severity"].value_counts().sort_index()
    axes[0, 1].bar(severity_counts.index, severity_counts.values, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Average Severity")
    axes[0, 1].set_ylabel("Number of Clusters")
    axes[0, 1].set_title("Average Severity Distribution")

    # 3. Defect category distribution (top 15)
    top_categories = stats["defect_category"].value_counts().head(15)
    axes[1, 0].barh(range(len(top_categories)), top_categories.values, edgecolor="black", alpha=0.7)
    axes[1, 0].set_yticks(range(len(top_categories)))
    axes[1, 0].set_yticklabels(top_categories.index)
    axes[1, 0].set_xlabel("Number of Clusters")
    axes[1, 0].set_title("Top 15 Defect Categories")
    axes[1, 0].invert_yaxis()

    # 4. Safety-relevant vs total
    safety_relevant = stats["is_safety_relevant"].sum()
    total = len(stats)
    axes[1, 1].pie([safety_relevant, total - safety_relevant],
                   labels=[f"Safety-relevant ({safety_relevant})", f"Other ({total - safety_relevant})"],
                   colors=["#e74c3c", "#3498db"])
    axes[1, 1].set_title("Clusters by Safety Relevance")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()


def visualize_temporal_intensity(
    cluster: DefectCluster,
    output_path: str | Path | None = None,
) -> None:
    """Visualize temporal intensity for a single cluster.

    Args:
        cluster: DefectCluster object
        output_path: Output image path (None to show)
    """
    import matplotlib.pyplot as plt

    if not cluster.monthly_intensity:
        logger.warning("No temporal intensity data for this cluster")
        return

    months = sorted(cluster.monthly_intensity.keys())
    intensities = [cluster.monthly_intensity[m] for m in months]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(months, intensities, marker="o", markersize=3, linewidth=1)
    ax.fill_between(range(len(months)), intensities, alpha=0.3)
    ax.set_xlabel("Month")
    ax.set_ylabel("Complaint Count")
    ax.set_title(f"Temporal Intensity: Cluster {cluster.cluster_id}\n"
                 f"{cluster.defect_category} (n={cluster.size}, avg severity={cluster.avg_severity:.1f})")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
