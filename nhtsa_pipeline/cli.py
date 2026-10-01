"""
NHTSA Complaint Defect Detection Pipeline
==========================================

CLI entry point for the full pipeline:
    1. Download & preprocess complaint data
    2. Extract defect annotations with LLM
    3. Cluster defect signals
    4. Analyze recall gaps

Usage:
    # Full pipeline
    python -m nhtsa_pipeline.cli --data-dir data/ --output-dir results/

    # Step 1: Download data
    python -m nhtsa_pipeline.cli download --data-dir data/

    # Step 2: Extract annotations
    python -m nhtsa_pipeline.cli extract \
        --input data/complaints.parquet \
        --output data/annotations.parquet \
        --model qwen/qwen3.6-35b-a3b

    # Step 3: Cluster
    python -m nhtsa_pipeline.cli cluster \
        --input data/annotations.parquet \
        --output data/clusters.parquet \
        --min-size 50

    # Step 4: Gap analysis
    python -m nhtsa_pipeline.cli gaps \
        --clusters data/clusters.json \
        --recalls data/recalls.csv \
        --output results/gaps.json

    # Status
    python -m nhtsa_pipeline.cli status --output-dir results/

    # Visualize
    python -m nhtsa_pipeline.cli visualize --clusters data/clusters.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nhtsa_pipeline.data import (
    download_complaints,
    download_recalls,
    load_all_complaints,
    load_all_recalls,
    load_from_parquet,
    export_to_parquet,
    get_narrative_stats,
    make_gold_label_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nhtsa_pipeline")


def cmd_download(args: argparse.Namespace) -> None:
    """Download official NHTSA flat-file data."""
    downloaded = []
    if args.dataset in {"complaints", "all"}:
        logger.info("Downloading NHTSA complaint data...")
        downloaded.extend(download_complaints(args.data_dir, year_range=args.year_range))
    if args.dataset in {"recalls", "all"}:
        logger.info("Downloading NHTSA recall data...")
        downloaded.extend(download_recalls(args.data_dir, recall_range=args.recall_range))
    logger.info(f"Downloaded/available {len(downloaded)} files: {[f.name for f in downloaded]}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess downloaded NHTSA flat files into Parquet."""
    if args.dataset == "complaints":
        logger.info("Preprocessing complaint data...")
        df = load_all_complaints(args.data_dir, year_ranges=args.year_ranges, received_year=args.received_year)
        stats = get_narrative_stats(df)
        logger.info(f"Narrative stats:\n{stats.to_string()}")
    elif args.dataset == "recalls":
        logger.info("Preprocessing recall data...")
        df = load_all_recalls(args.data_dir, recall_range=args.recall_range)
    else:
        raise ValueError(f"Unsupported preprocess dataset: {args.dataset}")

    export_to_parquet(df, args.output)
    logger.info(f"Preprocessed {len(df):,} {args.dataset} rows -> {args.output}")


def cmd_sample(args: argparse.Namespace) -> None:
    """Create a stratified manual gold-label annotation sample."""
    logger.info("Loading complaints for gold-label sample...")
    df = load_from_parquet(args.input)
    sample = make_gold_label_sample(
        df,
        n=args.n,
        random_state=args.random_state,
        min_per_priority_group=args.min_per_priority_group,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)

    manifest = {
        "input": str(args.input),
        "output": str(output),
        "sample_size": int(len(sample)),
        "random_state": args.random_state,
        "min_per_priority_group": args.min_per_priority_group,
        "stratum_counts": sample["_sample_stratum"].value_counts().to_dict()
        if "_sample_stratum" in sample.columns else {},
        "component_counts_top20": sample["Component"].value_counts().head(20).to_dict()
        if "Component" in sample.columns else {},
        "label_columns": [c for c in sample.columns if c.startswith("gold_")] + ["reviewer_id"],
        "note": "Manual labels are blank by design. Do not treat this CSV as evaluated data until reviewed.",
    }
    manifest_path = output.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote gold-label sample: %s", output)
    logger.info("Wrote sample manifest: %s", manifest_path)
    logger.info("Stratum counts: %s", manifest["stratum_counts"])


def cmd_extract(args: argparse.Namespace) -> None:
    """Run LLM-based defect extraction."""
    from nhtsa_pipeline.extract import extract_annotations

    logger.info("Loading complaint data...")
    df = load_from_parquet(args.input)

    logger.info(f"Loaded {len(df):,} complaints")
    logger.info(f"LLM endpoint: {args.base_url}")
    logger.info(f"Model: {args.model}")

    # Extract
    result_df = extract_annotations(
        df=df,
        output_path=args.output,
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
        resume=args.resume,
    )

    if not result_df.empty:
        # Save extraction quality report
        report = {
            "total_complaints": len(df),
            "extracted": len(result_df),
            "extraction_rate": len(result_df) / len(df) if len(df) > 0 else 0,
            "avg_confidence": float(result_df["confidence"].mean()),
            "defect_category_distribution": result_df["defect_category"].value_counts().to_dict(),
            "severity_distribution": result_df["severity"].value_counts().to_dict(),
        }
        report_path = Path(args.output).with_suffix(".json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Extraction report saved to {report_path}")


def cmd_cluster(args: argparse.Namespace) -> None:
    """Run defect signal clustering."""
    from nhtsa_pipeline.cluster import (
        compute_embeddings,
        cluster_defect_signals,
        analyze_temporal_patterns,
        export_clusters,
        visualize_cluster_distribution,
    )

    logger.info("Loading annotations...")
    df = load_from_parquet(args.input)

    logger.info("Computing embeddings...")
    embeddings = compute_embeddings(df, model_name=args.embedding_model)

    logger.info("Clustering...")
    clusters = cluster_defect_signals(
        df, embeddings,
        min_cluster_size=args.min_size,
        distance_threshold=args.threshold,
    )

    logger.info("Analyzing temporal patterns...")
    clusters = analyze_temporal_patterns(clusters, df)

    logger.info(f"Exporting {len(clusters)} clusters...")
    export_clusters(clusters, args.output)

    # Show summary
    from nhtsa_pipeline.cluster import get_cluster_statistics
    stats = get_cluster_statistics(clusters)
    logger.info(f"Cluster summary:\n{stats.describe()}")

    # Visualize
    if args.visualize:
        visualize_cluster_distribution(clusters, Path(args.output).with_suffix(".png"))


def cmd_gaps(args: argparse.Namespace) -> None:
    """Run recall gap analysis."""
    from nhtsa_pipeline.gap_analysis import (
        load_recalls,
        identify_gaps,
        export_gaps,
        visualize_gaps,
    )

    logger.info("Loading clusters...")
    with open(args.clusters) as f:
        cluster_data = json.load(f)

    # Reconstruct cluster objects
    clusters = []
    for c in cluster_data:
        cluster = type("Cluster", (), {
            "cluster_id": c["cluster_id"],
            "complaint_ids": c["complaint_ids"],
            "defect_category": c["defect_category"],
            "avg_severity": c["avg_severity"],
            "avg_confidence": c["avg_confidence"],
            "size": c["size"],
            "date_range": c["date_range"],
            "first_complaint_date": c["first_complaint_date"],
            "last_complaint_date": c["last_complaint_date"],
            "peak_month": c["peak_month"],
            "peak_intensity": c["peak_intensity"],
            "monthly_intensity": c["monthly_intensity"],
            "is_safety_relevant": c["is_safety_relevant"],
            "is_high_severity": c["is_high_severity"],
            "embedding": None,
        })()
        clusters.append(cluster)

    logger.info(f"Loaded {len(clusters)} clusters")

    logger.info("Loading recalls...")
    recalls = load_recalls(args.recalls)

    # Load annotations for temporal analysis
    logger.info("Loading annotations for temporal analysis...")
    annot_path = Path(args.clusters).parent / "annotations.parquet"
    if annot_path.exists():
        df = load_from_parquet(annot_path)
        for cluster in clusters:
            cluster.compute_temporal_intensity(df)

    logger.info("Identifying gaps...")
    gaps = identify_gaps(clusters, recalls, recalls)

    logger.info(f"Identified {len(gaps)} potential recall gaps")

    # Export
    export_gaps(gaps, args.output)

    # Visualize
    if args.visualize:
        visualize_gaps(gaps, Path(args.output).with_suffix(".png"))

    # Print summary
    priority_counts = {}
    for gap in gaps:
        priority_counts[gap.priority] = priority_counts.get(gap.priority, 0) + 1
    logger.info(f"Priority breakdown: {priority_counts}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show pipeline status."""
    output_dir = Path(args.output_dir)

    status = {
        "data": False,
        "annotations": False,
        "clusters": False,
        "gaps": False,
    }

    if (output_dir / "complaints.parquet").exists():
        status["data"] = True
        df = load_from_parquet(output_dir / "complaints.parquet")
        status["complaint_count"] = len(df)

    if (output_dir / "annotations.parquet").exists():
        status["annotations"] = True
        df = load_from_parquet(output_dir / "annotations.parquet")
        status["annotation_count"] = len(df)

    if (output_dir / "clusters.json").exists():
        status["clusters"] = True
        with open(output_dir / "clusters.json") as f:
            clusters = json.load(f)
        status["cluster_count"] = len(clusters)

    if (output_dir / "gaps.json").exists():
        status["gaps"] = True
        with open(output_dir / "gaps.json") as f:
            gaps = json.load(f)
        status["gap_count"] = len(gaps)

    print(json.dumps(status, indent=2))


def cmd_visualize(args: argparse.Namespace) -> None:
    """Generate visualizations."""
    clusters_path = Path(args.clusters)
    if not clusters_path.exists():
        logger.error(f"Clusters file not found: {clusters_path}")
        sys.exit(1)

    with open(clusters_path) as f:
        cluster_data = json.load(f)

    # Reconstruct cluster objects (simplified for visualization)
    class SimpleCluster:
        def __init__(self, data):
            self.cluster_id = data["cluster_id"]
            self.defect_category = data["defect_category"]
            self.avg_severity = data["avg_severity"]
            self.avg_confidence = data["avg_confidence"]
            self.size = data["size"]
            self.first_complaint_date = data["first_complaint_date"]
            self.last_complaint_date = data["last_complaint_date"]
            self.monthly_intensity = data.get("monthly_intensity", {})
            self.complaint_ids = data["complaint_ids"]
            self.is_safety_relevant = data["is_safety_relevant"]
            self.is_high_severity = data["is_high_severity"]
            self.peak_month = data.get("peak_month", "")
            self.peak_intensity = data.get("peak_intensity", 0)

    clusters = [SimpleCluster(c) for c in cluster_data]

    from nhtsa_pipeline.cluster import visualize_cluster_distribution
    visualize_cluster_distribution(clusters, args.output)


def main():
    parser = argparse.ArgumentParser(
        description="NHTSA Complaint Defect Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python -m nhtsa_pipeline.cli --data-dir data/ --output-dir results/

  # Download only
  python -m nhtsa_pipeline.cli download --data-dir data/

  # Extract annotations
  python -m nhtsa_pipeline.cli extract --input data/complaints.parquet --output data/annotations.parquet

  # Cluster and analyze
  python -m nhtsa_pipeline.cli cluster --input data/annotations.parquet --output data/clusters.parquet
        """,
    )

    parser.add_argument("--data-dir", default="data/", help="Directory for complaint data")
    parser.add_argument("--output-dir", default="results/", help="Directory for pipeline outputs")
    parser.add_argument("--year-range", help="Specific year range to download (e.g., 2020-2024)")
    parser.add_argument("--year-ranges", nargs="+", help="Year ranges to preprocess")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download
    p_download = subparsers.add_parser("download", help="Download official NHTSA flat-file data")
    p_download.add_argument("--data-dir", default="data/")
    p_download.add_argument("--dataset", choices=["complaints", "recalls", "all"], default="complaints")
    p_download.add_argument("--year-range", help="Complaint year block, e.g. 2015-2019 for the 2018 pilot")
    p_download.add_argument("--recall-range", choices=["post-2010", "pre-2010", "all"], default="post-2010")

    # Preprocess
    p_preprocess = subparsers.add_parser("preprocess", help="Preprocess official NHTSA flat files")
    p_preprocess.add_argument("--data-dir", default="data/")
    p_preprocess.add_argument("--dataset", choices=["complaints", "recalls"], default="complaints")
    p_preprocess.add_argument("--output", default="data/complaints.parquet")
    p_preprocess.add_argument("--year-ranges", nargs="+")
    p_preprocess.add_argument("--received-year", type=int, help="Filter complaints by NHTSA received year, e.g. 2018")
    p_preprocess.add_argument("--recall-range", choices=["post-2010", "pre-2010", "all"], default="post-2010")

    # Gold-label sample
    p_sample = subparsers.add_parser("sample", help="Create stratified CSV for manual gold-label annotation")
    p_sample.add_argument("--input", default="data/complaints_2018.parquet", help="Input complaints Parquet")
    p_sample.add_argument("--output", default="data/gold_sample_2018.csv", help="Output annotation CSV")
    p_sample.add_argument("--n", type=int, default=500, help="Sample size")
    p_sample.add_argument("--random-state", type=int, default=42)
    p_sample.add_argument("--min-per-priority-group", type=int, default=25)

    # Extract
    p_extract = subparsers.add_parser("extract", help="Extract defect annotations")
    p_extract.add_argument("--input", required=True, help="Input complaints Parquet")
    p_extract.add_argument("--output", required=True, help="Output annotations Parquet")
    p_extract.add_argument("--base-url", default="http://localhost:1234/v1")
    p_extract.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    p_extract.add_argument("--batch-size", type=int, default=1000)
    p_extract.add_argument("--resume", action="store_true", default=True)

    # Cluster
    p_cluster = subparsers.add_parser("cluster", help="Cluster defect signals")
    p_cluster.add_argument("--input", required=True, help="Input annotations Parquet")
    p_cluster.add_argument("--output", required=True, help="Output clusters path")
    p_cluster.add_argument("--min-size", type=int, default=50)
    p_cluster.add_argument("--threshold", type=float, default=0.45)
    p_cluster.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p_cluster.add_argument("--visualize", action="store_true")

    # Gaps
    p_gaps = subparsers.add_parser("gaps", help="Analyze recall gaps")
    p_gaps.add_argument("--clusters", required=True, help="Clusters JSON file")
    p_gaps.add_argument("--recalls", required=True, help="Recalls CSV/JSON file")
    p_gaps.add_argument("--output", required=True, help="Output gaps file")
    p_gaps.add_argument("--visualize", action="store_true")

    # Status
    p_status = subparsers.add_parser("status", help="Show pipeline status")
    p_status.add_argument("--output-dir", default="results/")

    # Visualize
    p_viz = subparsers.add_parser("visualize", help="Generate visualizations")
    p_viz.add_argument("--clusters", required=True, help="Clusters JSON file")
    p_viz.add_argument("--output", default="results/visualization.png")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Route to command handler
    commands = {
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "sample": cmd_sample,
        "extract": cmd_extract,
        "cluster": cmd_cluster,
        "gaps": cmd_gaps,
        "status": cmd_status,
        "visualize": cmd_visualize,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
