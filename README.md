# NHTSA Complaint Defect Detection Pipeline

Research code for studying whether large language models can turn NHTSA consumer-complaint narratives into structured defect signals for early safety triage.

The repository currently supports a 2018 pilot and is designed to scale to additional official NHTSA complaint and recall flat files. It is a research prototype: outputs should be audited before being treated as safety or regulatory conclusions.

## What the pipeline does

1. Downloads official NHTSA complaint and recall ZIP archives.
2. Parses the tab-delimited flat files and standardizes dates, identifiers, and common fields.
3. Exports complaint or recall data to Parquet for repeatable downstream processing.
4. Creates a stratified CSV sample for manual gold-label annotation.
5. Sends complaint narratives to an OpenAI-compatible LLM endpoint and validates structured defect annotations.
6. Embeds and clusters defect signals, then measures their temporal intensity.
7. Compares clusters with recall records to identify recall-gap candidates and detection lag.

The main research unit is intended to be make × model × model year × component × month. Narrative clusters are an interpretability layer, not proof that a defect exists or that a recall was missed.

## Repository layout

```text
nhtsa_pipeline/                 Pipeline package and CLI
  data.py                       NHTSA download, parsing, and Parquet utilities
  schema.py                     Defect annotation schema and extraction prompts
  extract.py                    OpenAI-compatible LLM extraction and checkpointing
  cluster.py                    Embeddings, clustering, temporal analysis, plots
  gap_analysis.py               Recall matching and gap-candidate analysis
  cli.py                        Command-line entry point
tests/                          Parser and sampling tests
outputs/                        Annotation rubric and research notes
paper.md                       Draft manuscript; reported results are unverified
pyproject.toml                  uv/project configuration
```

Downloaded data and generated results are intentionally not checked into the repository. Keep them under local `data/` and `results/` directories.

## Setup with uv

Install [uv](https://docs.astral.sh/uv/) if needed, then run from the repository root:

```bash
uv sync --extra dev
```

This creates or updates the project environment and lockfile. Run tests with:

```bash
uv run pytest
```

The CLI can be invoked either as a module or through the installed console script:

```bash
uv run python -m nhtsa_pipeline.cli --help
uv run nhtsa-pipeline --help
```

## Quick start: 2018 pilot

### 1. Download complaint and recall data

The 2018 pilot uses the official 2015–2019 complaint block and post-2010 recalls:

```bash
uv run nhtsa-pipeline download \
  --data-dir data/ \
  --dataset complaints \
  --year-range 2015-2019

uv run nhtsa-pipeline download \
  --data-dir data/ \
  --dataset recalls \
  --recall-range post-2010
```

### 2. Preprocess the flat files

```bash
uv run nhtsa-pipeline preprocess \
  --data-dir data/ \
  --dataset complaints \
  --year-ranges 2015-2019 \
  --received-year 2018 \
  --output data/complaints_2018.parquet

uv run nhtsa-pipeline preprocess \
  --data-dir data/ \
  --dataset recalls \
  --recall-range post-2010 \
  --output data/recalls_post_2010.parquet
```

Complaint rows retain the official `CMPLID` and `ODINO` identifiers. Standardized columns include `Complaint ID`, `ODI Number`, `Make`, `Model`, `Model Year`, `Date Received`, `Component`, and `Narrative`.

### 3. Create a manual annotation sample

```bash
uv run nhtsa-pipeline sample \
  --input data/complaints_2018.parquet \
  --output data/gold_sample_2018.csv \
  --n 500
```

The sample over-represents records with crash, fire, injury, or death indicators and common components. The generated `.manifest.json` records the sampling settings. Complete the blank `gold_*` columns using [the annotation rubric](outputs/gold_label_annotation_rubric.md).

### 4. Extract LLM annotations

`extract` expects an OpenAI-compatible `/models` and `/chat/completions` endpoint. By default it targets LM Studio at `http://localhost:1234/v1`.

```bash
uv run nhtsa-pipeline extract \
  --input data/complaints_2018.parquet \
  --output data/annotations_2018.parquet \
  --base-url http://localhost:1234/v1 \
  --model qwen/qwen3.6-35b-a3b \
  --batch-size 1000
```

Extraction is checkpointed in the output Parquet file, so rerunning the command resumes from previously processed complaint IDs. The command also writes an extraction report beside the output file.

### 5. Cluster defect signals

```bash
uv run nhtsa-pipeline cluster \
  --input data/annotations_2018.parquet \
  --output data/clusters_2018 \
  --min-size 50 \
  --embedding-model all-MiniLM-L6-v2 \
  --visualize
```

This writes JSON cluster metadata, tabular cluster statistics in Parquet, and—when requested—a PNG distribution plot. The `--output` value is a base path; the implementation adds `.json` and `.parquet`.

### 6. Analyze recall-gap candidates

```bash
uv run nhtsa-pipeline gaps \
  --clusters data/clusters_2018.json \
  --recalls data/recalls_post_2010.parquet \
  --output results/gaps_2018.json \
  --visualize
```

Treat unmatched clusters as candidates for review. They are not evidence of a missed recall without additional validation such as investigation records, technical service bulletins, manufacturer communications, or expert review.

## Useful commands

```bash
# Show which standard outputs exist
uv run nhtsa-pipeline status --output-dir results/

# Render a cluster visualization from an existing JSON file
uv run nhtsa-pipeline visualize \
  --clusters data/clusters_2018.json \
  --output results/clusters_2018.png
```

For command-specific options, run `uv run nhtsa-pipeline <command> --help`.

## Data sources

The downloader uses NHTSA’s official ODI flat files:

- Complaints: `https://static.nhtsa.gov/odi/ffdd/cmpl/`
- Recalls: `https://static.nhtsa.gov/odi/ffdd/rcl/`

The repository does not redistribute downloaded complaint narratives or recall records. Check NHTSA’s terms and the applicable data documentation before publishing derived data.

## Research status and limitations

- This is an experimental research pipeline, not a production safety-monitoring system.
- Complaint narratives are self-reported, noisy, and incomplete.
- LLM annotations require human evaluation; confidence scores are model outputs, not calibrated probabilities.
- The gold-label sample is intentionally stratified for triage evaluation and should not be used as a population estimate without weighting.
- Clustering is currently based on sentence-transformer embeddings and hierarchical clustering; large-scale runs may require more scalable aggregation or clustering methods.
- `paper.md` contains a draft framing and illustrative result values. Do not cite its numerical results as empirical findings until the experiments are run and documented.

See [the literature survey](outputs/north_star_literature_survey.md) for the recommended research question, baselines, and definitions of “recall-gap candidate.”
