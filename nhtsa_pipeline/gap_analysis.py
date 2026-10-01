"""
Recall gap analysis.

Implements:
- Matching defect clusters to NHTSA recalls
- Detection lag calculation
- Gap identification and classification
- Gap prioritization
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from nhtsa_pipeline.cluster import DefectCluster

logger = logging.getLogger(__name__)


class RecallGap:
    """Represents a potential recall gap."""

    class Priority(str):
        HIGH = "high-priority"
        MODERATE = "moderate-priority"
        LOW = "low-priority"
        EMERGING = "emerging-signal"

    def __init__(
        self,
        cluster: DefectCluster,
        gap_type: str,
        priority: str,
        detection_lag_months: float | None,
        sustained_high_months: int,
        severity_score: float,
        explanation: str = "",
        verified: bool = False,
        expert_notes: str = "",
    ):
        self.cluster = cluster
        self.gap_type = gap_type
        self.priority = priority
        self.detection_lag_months = detection_lag_months
        self.sustained_high_months = sustained_high_months
        self.severity_score = severity_score
        self.explanation = explanation
        self.verified = verified
        self.expert_notes = expert_notes

    @property
    def is_critical(self) -> bool:
        return self.priority == self.Priority.HIGH

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster.cluster_id,
            "defect_category": self.cluster.defect_category,
            "cluster_size": self.cluster.size,
            "avg_severity": self.cluster.avg_severity,
            "priority": self.priority,
            "gap_type": self.gap_type,
            "detection_lag_months": self.detection_lag_months,
            "sustained_high_months": self.sustained_high_months,
            "severity_score": round(self.severity_score, 2),
            "first_complaint": self.cluster.first_complaint_date,
            "last_complaint": self.cluster.last_complaint_date,
            "peak_month": self.cluster.peak_month,
            "peak_intensity": self.cluster.peak_intensity,
            "explanation": self.explanation,
            "verified": self.verified,
            "expert_notes": self.expert_notes,
        }


def load_recalls(recalls_path: str | Path) -> pd.DataFrame:
    """Load NHTSA recall data.

    Expected columns:
    - Recall Number
    - Make
    - Model
    - Model Year
    - Component
    - Defect Summary
    - Reported Date
    - Manufacture Date Range

    Args:
        recalls_path: Path to recalls CSV/JSON

    Returns:
        DataFrame of recalls
    """
    recalls_path = Path(recalls_path)

    if recalls_path.suffix == ".json":
        recalls = pd.read_json(recalls_path)
    else:
        recalls = pd.read_csv(recalls_path)

    # Standardize column names
    col_map = {}
    for expected in ["Recall Number", "Make", "Model", "Model Year", "Component",
                      "Defect Summary", "Reported Date", "Manufacture Date Range"]:
        for col in recalls.columns:
            if expected.lower() in col.lower():
                col_map[col] = expected
                break

    recalls = recalls.rename(columns=col_map)

    # Parse dates
    if "Reported Date" in recalls.columns:
        recalls["Reported Date"] = pd.to_datetime(recalls["Reported Date"], errors="coerce")

    # Clean model year
    if "Model Year" in recalls.columns:
        recalls["Model Year"] = pd.to_numeric(recalls["Model Year"], errors="coerce")

    logger.info(f"Loaded {len(recalls):,} recalls")
    return recalls


def match_cluster_to_recall(
    cluster: DefectCluster,
    recalls: pd.DataFrame,
    recalls_df: pd.DataFrame,
) -> tuple[bool, pd.DataFrame | None]:
    """Check if a defect cluster corresponds to a known recall.

    Matching criteria:
    1. Same make/model/year range
    2. Same or related defect category
    3. Recall reported after cluster's first complaint

    Args:
        cluster: DefectCluster to check
        recalls: Raw recalls DataFrame
        recalls_df: Processed recalls (for component mapping)

    Returns:
        (has_match, matching_recall)
    """
    # Get cluster's make/model/year range from complaint data
    # (In practice, this would come from the cluster's complaint metadata)
    # For now, use defect category as proxy

    # Filter recalls by defect category similarity
    category = cluster.defect_category.lower()

    # Map defect categories to recall component keywords
    component_keywords = {
        "brake_system_failure": ["brake"],
        "steering_defect": ["steering"],
        "suspension_defect": ["suspension", "shock", "strut"],
        "tire_wheel_defect": ["tire", "wheel"],
        "engine_stalling_shutdown": ["engine", "stall"],
        "transmission_failure": ["transmission"],
        "engine_performance_degradation": ["engine"],
        "fuel_system_defect": ["fuel"],
        "exhaust_emissions_defect": ["exhaust", "emission"],
        "electrical_system_failure": ["electrical", "wiring", "short"],
        "battery_charging_defect": ["battery", "charging", "alternator"],
        "adas_sensor_failure": ["camera", "sensor", "radar", "lidar", "adas"],
        "autonomous_driving_defect": ["autopilot", "autonomous", "self-driving"],
        "body_trim_defect": ["body", "trim", "paint"],
        "structural_corrosion_defect": ["corrosion", "rust", "frame"],
        "glass_window_defect": ["glass", "window", "windshield"],
        "seat_restraint_defect": ["seat", "airbag", "restraint", "belt"],
        "hvac_defect": ["hvac", "heater", "air conditioning", "climate"],
    }

    keywords = component_keywords.get(category, [category])

    # Search recalls for matching component
    matching = []
    for _, recall in recalls.iterrows():
        defect_summary = str(recall.get("Defect Summary", "")).lower()
        component = str(recall.get("Component", "")).lower()

        # Check if any keyword matches recall description or component
        if any(kw in defect_summary or kw in component for kw in keywords):
            matching.append(recall)

    if not matching:
        return False, None

    # Check temporal: recall reported after cluster's first complaint
    cluster_first = pd.to_datetime(cluster.first_complaint_date)
    for recall in matching:
        recall_date = recall.get("Reported Date")
        if pd.notna(recall_date) and recall_date >= cluster_first:
            return True, recall

    return False, None


def compute_detection_lag(
    cluster: DefectCluster,
    recall: pd.DataFrame,
) -> float:
    """Compute detection lag in months.

    Lag = recall reported date - first complaint in cluster

    Args:
        cluster: DefectCluster
        recall: Matching recall record

    Returns:
        Detection lag in months
    """
    cluster_first = pd.to_datetime(cluster.first_complaint_date)
    recall_date = pd.to_datetime(recall["Reported Date"])

    lag_days = (recall_date - cluster_first).days
    lag_months = lag_days / 30.44  # Average days per month

    return max(0, lag_months)


def classify_gap_priority(
    cluster: DefectCluster,
    sustained_high_months: int,
    months_since_last_complaint: int,
) -> str:
    """Classify gap priority based on temporal and severity patterns.

    Priority levels:
    - HIGH: Sustained high intensity, severity >= 4, no recall after 24+ months
    - MODERATE: Moderate intensity, severity >= 3, no recall after 18+ months
    - LOW: Low intensity, severity >= 3, no recall after 12+ months
    - EMERGING: Recent onset (<12 months), rapid intensity growth

    Args:
        cluster: DefectCluster
        sustained_high_months: Months with sustained high intensity
        months_since_last_complaint: Months since last complaint in cluster

    Returns:
        Priority level string
    """
    if cluster.is_high_severity and sustained_high_months >= 6 and months_since_last_complaint >= 24:
        return RecallGap.Priority.HIGH
    elif cluster.is_safety_relevant and sustained_high_months >= 4 and months_since_last_complaint >= 18:
        return RecallGap.Priority.MODERATE
    elif cluster.is_safety_relevant and sustained_high_months >= 2 and months_since_last_complaint >= 12:
        return RecallGap.Priority.LOW
    elif months_since_last_complaint < 12 and cluster.is_safety_relevant:
        # Check for rapid growth
        intensities = list(cluster.monthly_intensity.values())
        if len(intensities) >= 3:
            recent = intensities[-3:]
            if recent[-1] > recent[0] * 2:  # Doubled in recent months
                return RecallGap.Priority.EMERGING

    return RecallGap.Priority.LOW


def identify_gaps(
    clusters: list[DefectCluster],
    recalls: pd.DataFrame,
    recalls_df: pd.DataFrame,
) -> list[RecallGap]:
    """Identify potential recall gaps from defect clusters.

    For each cluster, checks if a corresponding recall exists.
    If not, classifies the gap and computes detection lag.

    Args:
        clusters: List of DefectCluster objects
        recalls: Raw recalls DataFrame
        recalls_df: Processed recalls

    Returns:
        List of RecallGap objects
    """
    logger.info(f"Analyzing {len(clusters)} clusters for recall gaps...")

    gaps = []
    matched_count = 0

    for cluster in tqdm(clusters, desc="Gap analysis"):
        # Check if cluster corresponds to a recall
        has_match, matching_recall = match_cluster_to_recall(cluster, recalls, recalls_df)

        if has_match:
            matched_count += 1
            continue  # Not a gap

        # This is a potential gap
        sustained_high = sum(1 for v in cluster.monthly_intensity.values() if v >= 20)
        months_since_last = _months_between(cluster.last_complaint_date, "2024-12-31")

        priority = classify_gap_priority(cluster, sustained_high, months_since_last)

        # Compute severity score (0-100)
        severity_score = (
            cluster.avg_severity * 20 +  # Up to 100 for severity
            min(sustained_high * 5, 20) +  # Up to 20 for sustained intensity
            min(cluster.size / 100, 10)  # Up to 10 for cluster size
        )

        explanation = _generate_gap_explanation(cluster, sustained_high, priority)

        gap = RecallGap(
            cluster=cluster,
            gap_type="no_recall_found",
            priority=priority,
            detection_lag_months=None,
            sustained_high_months=sustained_high,
            severity_score=severity_score,
            explanation=explanation,
        )
        gaps.append(gap)

    logger.info(f"Matched {matched_count} clusters to recalls")
    logger.info(f"Identified {len(gaps)} potential recall gaps")

    # Sort by priority and severity score
    priority_order = {
        RecallGap.Priority.HIGH: 0,
        RecallGap.Priority.MODERATE: 1,
        RecallGap.Priority.LOW: 2,
        RecallGap.Priority.EMERGING: 3,
    }
    gaps.sort(key=lambda g: (priority_order.get(g.priority, 4), -g.severity_score))

    return gaps


def _generate_gap_explanation(
    cluster: DefectCluster,
    sustained_high_months: int,
    priority: str,
) -> str:
    """Generate human-readable explanation for the gap."""
    parts = [
        f"Cluster of {cluster.size:,} complaints about {cluster.defect_category}",
        f"avg severity {cluster.avg_severity:.1f}",
        f"sustained high intensity for {sustained_high_months} months",
    ]

    if cluster.is_high_severity:
        parts.append("HIGH SEVERITY: crash/catastrophic risk indicated")

    if cluster.avg_confidence < 0.8:
        parts.append("lower extraction confidence")

    return ". ".join(parts) + "."


def _months_between(date1: str, date2: str) -> int:
    """Calculate months between two date strings."""
    d1 = pd.to_datetime(date1)
    d2 = pd.to_datetime(date2)
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def export_gaps(
    gaps: list[RecallGap],
    output_path: str | Path,
) -> None:
    """Export gap analysis results.

    Args:
        gaps: List of RecallGap objects
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [g.to_dict() for g in gaps]

    # JSON export
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Summary statistics
    summary = {
        "total_clusters": len(gaps),
        "by_priority": {},
        "by_category": {},
        "avg_severity_score": 0,
        "high_severity_gaps": 0,
    }

    for gap in gaps:
        summary["by_priority"][gap.priority] = summary["by_priority"].get(gap.priority, 0) + 1
        summary["by_category"][gap.cluster.defect_category] = \
            summary["by_category"].get(gap.cluster.defect_category, 0) + 1
        summary["avg_severity_score"] += gap.severity_score
        if gap.cluster.is_high_severity:
            summary["high_severity_gaps"] += 1

    summary["avg_severity_score"] /= len(gaps) if gaps else 1

    with open(output_path.with_suffix("_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Exported {len(gaps)} gaps to {output_path}")
    logger.info(f"Priority breakdown: {summary['by_priority']}")


def visualize_gaps(
    gaps: list[RecallGap],
    output_path: str | Path | None = None,
) -> None:
    """Visualize gap analysis results.

    Args:
        gaps: List of RecallGap objects
        output_path: Output image path
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not gaps:
        logger.warning("No gaps to visualize")
        return

    # Priority distribution
    priority_counts = {}
    for gap in gaps:
        priority_counts[gap.priority] = priority_counts.get(gap.priority, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Priority pie chart
    colors = {"high-priority": "#e74c3c", "moderate-priority": "#f39c12",
              "low-priority": "#3498db", "emerging-signal": "#9b59b6"}
    axes[0].pie(priority_counts.values(), labels=priority_counts.keys(),
                colors=[colors.get(k, "#95a5a6") for k in priority_counts.keys()],
                autopct="%1.1f%%")
    axes[0].set_title("Recall Gap Priority Distribution")

    # 2. Severity score histogram
    axes[1].hist([g.severity_score for g in gaps], bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Severity Score")
    axes[1].set_ylabel("Number of Gaps")
    axes[1].set_title("Gap Severity Score Distribution")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved gap visualization to {output_path}")
    else:
        plt.show()
