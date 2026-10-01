# North Star Literature Survey: LLM-Assisted Early Vehicle Defect Detection from NHTSA Complaints

Date: 2026-09-30  
Repository: `nhtsa-paper`  
Immediate objective: refine the research paper idea before implementation.  
Initial empirical scope: 2018 NHTSA complaints, with design expandable to 2010-present.  
Inference target: local LM Studio / OpenAI-compatible endpoint.  
Status: literature-grounded concept document; manuscript numbers are treated as placeholders until experiments are run.

## 1. North star

Build a reproducible research pipeline that asks:

> Can structured LLM extraction over NHTSA consumer complaint narratives improve early triage of vehicle defect signals relative to simpler complaint-count, keyword, topic-model, and embedding baselines?

The strongest paper idea is **not** “LLMs find missed recalls automatically.” A defensible version is:

> NHTSA complaints are noisy, self-reported early signals. LLMs may help convert unstructured narratives into structured defect-event records—component, failure mode, severity, safety outcome, time, and vehicle context—which can then be aggregated into interpretable make-model-year/component time series for analyst triage and comparison against investigations/recalls.

This framing avoids overstating causality or regulatory ground truth while preserving a clear contribution.

## 2. What the literature says

### 2.1 NHTSA complaints are a valid but incomplete signal source

NHTSA’s official complaint flat file contains safety-related defect complaints received since January 1, 1995. Records are tab-delimited, dates use `YYYYMMDD`, and the key free-text field is `CDESCR`; useful structured fields include `CMPLID`, `ODINO`, `MAKETXT`, `MODELTXT`, `YEARTXT`, `COMPDESC`, `CRASH`, `FIRE`, `INJURED`, `DEATHS`, `FAILDATE`, and `LDATE`.

NHTSA recalls are also official tab-delimited flat files. Recall records include `CAMPNO`, make/model/model-year, component, affected population, report date `RCDATE`, `DESC_DEFECT`, consequence, and remedy fields.

Implication: the project should use complaints as **signals**, not as standalone proof of defect existence. Recalls and investigations should be treated as delayed, partial outcome labels.

### 2.2 Prior work already mined NHTSA complaint text

Ghazizadeh, McDonald, and Lee (2014) used latent semantic analysis and hierarchical clustering on NHTSA owner complaint narratives, showing that free-response complaints contain discoverable patterns and time trends. This is the closest classic baseline for the proposed work.

Implication: novelty cannot be “first use of text mining on NHTSA complaints.” A better novelty claim is structured LLM extraction plus explicit comparison against classical NLP and outcome-linked recall/investigation timelines.

### 2.3 Automotive recall prediction has used user-generated text and structured features

Zhang et al. (2015) studied vehicle recall prediction from user-generated online vehicle text plus official recall data using text-mining and classical classifiers.

Li et al. (2025) developed an XGBoost-based recall-risk model using NHTSA complaints and recall records, reporting strong AUC across future prediction windows according to the abstract/metadata available from PubMed and Wiley.

Implication: a credible evaluation needs baselines: complaint counts, component counts, keyword severity flags, topic models, embedding clusters, and structured ML models. LLM extraction should be tested for incremental value, not assumed to be superior.

### 2.4 Topic modeling for recall management is directly adjacent

Shi, Qu, and Liu’s MSI working paper, “Screening Consumer Complaints for Recall Management: A Topic Model for Decision Automation,” uses NHTSA complaints and recalls to extract defect topics and predict recall-related outcomes.

Implication: the paper should cite this as a central comparator. The project can differentiate by producing human-interpretable structured event records and by evaluating whether LLM-derived labels improve lead-time triage.

### 2.5 Recent NLP systems support defect-investigation workflows

Francis, Cates, and Caldiera (2023), “SmarTxT,” proposed an NLP system for efficient vehicle defect investigation using NHTSA-related defect texts. This supports a workflow-oriented framing: retrieval, grouping, and analyst support rather than autonomous regulatory decision-making.

Implication: the paper should emphasize decision support, not automated recall declaration.

### 2.6 Language-model adaptation to ODI complaints has precedent

An arXiv paper, “BERT as a Knowledge Base for Automotive Complaint Analysis” (2020, arXiv:2012.02558), continued language-model pretraining on ODI complaints and probed whether domain-specific automotive knowledge was captured.

Implication: language models are plausible for this domain, but the proposed work should still measure extraction accuracy on a manually reviewed sample.

## 3. Recommended research question

Primary research question:

> For 2018 NHTSA complaints, do LLM-extracted structured defect-event records improve early identification and characterization of later recall- or investigation-linked vehicle defect signals compared with non-LLM baselines?

Subquestions:

1. **Extraction validity:** Can a local LLM reliably extract component, failure mode, severity, safety outcome, and symptom fields from complaint narratives?
2. **Aggregation unit:** Which aggregation level best balances specificity and statistical power: complaint row, ODI number, make-model-year-component-month, or discovered cluster?
3. **Signal detection:** Do LLM-derived labels improve precision/recall or lead time for signals that later map to recalls/investigations?
4. **Interpretability:** Are LLM-derived clusters easier to audit than topic-model or embedding-only clusters?
5. **Failure modes:** Where do LLMs hallucinate, overgeneralize, or miss low-frequency but safety-critical defects?

## 4. Recommended unit of analysis

Because the user is unsure, the recommended design is hierarchical:

1. **Raw record:** preserve every NHTSA complaint flat-file row, keyed by `CMPLID`.
2. **Complaint event:** group rows by `ODINO` only for analyses where multiple component rows describe the same consumer complaint. Do not blindly deduplicate before preserving component rows, because NHTSA notes `ODINO` may repeat for multiple components.
3. **Signal cell:** aggregate to **make × model × model-year × component × month**. This should be the main statistical unit for early warning.
4. **Narrative cluster:** optional layer within a signal cell, using text/LLM failure-mode labels to split broad components into interpretable defect themes.

Recommended first paper scope: use 2018 complaints as input and link them to recalls/investigations from 2018 onward, while retaining complaint history from 2010-present in the eventual design.

## 5. Recommended definition of “recall gap”

Avoid the phrase “missed recall” unless externally validated. Use three tiers:

- **Unmatched signal:** high-volume or high-severity complaint cluster with no matching recall in the comparison window.
- **Delayed-match signal:** complaint signal that appears before a matching recall or investigation, enabling lead-time analysis.
- **Candidate recall-gap signal:** unmatched signal that also has corroboration from investigations, TSBs/manufacturer communications, crash/fire/injury/death flags, or expert review.

For the north-star paper, the safest construct is **“recall-gap candidate”**, not “recall gap.”

## 6. Recommended evaluation for the 2018 pilot

### 6.1 Data

Download locally and keep out of git:

- 2018 complaint records from official NHTSA complaint bulk flat files.
- Post-2010 recall flat file.
- Optionally: defect investigations flat file for stronger outcome linkage.
- Optional API calls for spot checks by vehicle or campaign number.

`.gitignore` should exclude raw ZIP/TXT/Parquet outputs and LLM extraction outputs if they contain complaint narratives.

### 6.2 Gold sample

Create a manually reviewed sample before running large-scale claims:

- Stratify 300–500 complaints from 2018 by component, crash/fire/injury flags, and narrative length.
- Have a human label component/failure mode/severity/safety outcome.
- Compare local LM Studio extraction against labels.

Minimum metrics:

- exact/partial match for component and subsystem,
- severity agreement,
- safety-outcome flag precision/recall,
- JSON/schema validity rate,
- hallucination/error categories.

### 6.3 Baselines

Compare LLM-derived signals against:

1. Complaint-count baseline by make/model/year/component/month.
2. Keyword/severity lexicon baseline.
3. TF-IDF + clustering or classifier baseline.
4. Topic model baseline, motivated by prior recall-management literature.
5. Sentence-transformer embedding clustering baseline.
6. XGBoost/structured feature baseline if outcome labels are available.

### 6.4 Outcome labels

Use outcome tiers rather than a single binary label:

- recall match by make/model/model-year/component and recall text similarity,
- investigation match if available,
- no official match within the observation window,
- ambiguous/manual-review-needed.

## 7. Proposed paper contribution

A defensible contribution statement:

> We introduce and evaluate a reproducible LLM-assisted pipeline for transforming NHTSA complaint narratives into structured defect-event signals. On a 2018 pilot dataset, we compare LLM-derived signals with classical text-mining and structured complaint baselines, evaluate extraction accuracy on a human-labeled sample, and analyze whether structured narratives improve interpretable early triage of later recall- or investigation-linked defects.

What not to claim yet:

- Do not claim a specific number of clusters, gaps, or percentage improvements until experiments are run.
- Do not claim that unmatched complaint clusters are actual missed recalls.
- Do not claim novelty over all NHTSA complaint text mining.

## 8. Implementation implications for the existing repository

Before experiments, the code needs concept-aligned fixes:

1. Replace current complaint URL/parser assumptions with official tab-delimited NHTSA flat-file parsing.
2. Add official recall flat-file ingestion and field mapping.
3. Add missing dependencies such as `httpx` and `pyarrow`.
4. Fix export path bugs in gap analysis.
5. Preserve `CMPLID` and `ODINO` separately.
6. Make 2018 a first-class pilot mode.
7. Add small fixtures and tests before full-data runs.
8. Replace full pairwise agglomerative clustering for large data; use cell-level aggregation and scalable clustering.
9. Treat manuscript numbers as placeholders and mark them clearly.

## 9. Open decisions

These decisions remain but no longer block the north-star idea:

1. **Taxonomy size:** keep the current 22-category taxonomy for the pilot, then revise based on gold-sample error analysis. Do not force 24 categories solely because the draft says 24.
2. **Recall-gap definition:** use “recall-gap candidate” tiers above.
3. **Unit of analysis:** preserve raw rows, analyze primarily at make-model-year-component-month, and use narrative clusters as an interpretability layer.
4. **2010-present scope:** design the pipeline for 2010-present, but run the first pilot on 2018.

## 10. Recommended next step

Write a short revised abstract and methods sketch around this north-star claim, then implement only the minimum data pipeline needed to download and parse 2018 complaints plus post-2010 recalls. Do not implement full LLM extraction until a gold-label protocol and schema are frozen.

## Sources

- NHTSA ODI Complaints data dictionary, `CMPL.txt`: https://static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt
- NHTSA ODI Recalls data dictionary, `RCL.txt`: https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt
- NHTSA Datasets and APIs: https://www.nhtsa.gov/nhtsa-datasets-and-apis
- NHTSA vPIC API documentation: https://vpic.nhtsa.dot.gov/api/
- Ghazizadeh, McDonald & Lee, “Text Mining to Decipher Free-Response Consumer Complaints,” 2014, DOI: https://doi.org/10.1177/0018720813519473; PubMed: https://pubmed.ncbi.nlm.nih.gov/25277026/
- Francis, Cates & Caldiera, “SmarTxT: A Natural Language Processing Approach for Efficient Vehicle Defect Investigation,” 2023, DOI: https://doi.org/10.1177/03611981221125744
- Zhang et al., “Predicting Vehicle Recalls with User-Generated Contents: A Text Mining Approach,” 2015, DOI: https://doi.org/10.1007/978-3-319-18455-5_3
- Shi, Qu & Liu, “Screening Consumer Complaints for Recall Management: A Topic Model for Decision Automation,” MSI Working Paper 24-102: https://thearf-org-unified-admin.s3.amazonaws.com/MSI_Report_24-102.pdf
- Li et al., “XGBoost-based risk prediction model for massive vehicle recalls using consumer complaints,” 2025, DOI: https://doi.org/10.1111/risa.70052; PubMed: https://pubmed.ncbi.nlm.nih.gov/40441680/
- “BERT as a Knowledge Base for Automotive Complaint Analysis,” 2020, arXiv:2012.02558: https://arxiv.org/abs/2012.02558
- pandas `DataFrame.to_parquet` docs: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html
- scikit-learn `AgglomerativeClustering` docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
