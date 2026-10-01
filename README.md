# Research Paper Ideas: NHTSA APIs + LLMs

## Dataset Inventory

### NHTSA Public APIs (no auth required)

| API | Endpoint | Data | Time Range |
|-----|----------|------|------------|
| **vPIC** | `vpic.nhtsa.dot.gov/api` | VIN decode, vehicle specs, ADAS/ADS tech presence, manufacturer data | 1981+ |
| **Crash/FARS** | `crashviewer.nhtsa.dot.gov/CrashAPI` | Fatal crashes, person/vehicle/weather/drug/distraction data | 2010+ |
| **NCAP** | `api.nhtsa.gov/safetyratings` | Crash test star ratings, crash avoidance test results | Varies |
| **ODI Recalls** | `api.nhtsa.gov/recalls` | Safety recalls, defect investigations | 1966+ |
| **Complaints** | `api.nhtsa.gov/complaints` | Consumer complaints with free-text narratives | 1995+ |
| **SGO 2021-01** | `nhtsa.gov/ODI/ADS/` | Autonomous vehicle incident reports with narrative descriptions | 2022+ |

### Downloadable Datasets (bulk CSV/ZIP)

- **Complaints** — 36-78 MB per 5-year block, free-text narrative fields
- **Recalls** — RCL (Recall Communications), FLAT_RCL (annual/quarterly reports)
- **TSBS** — Technical Service Bulletins (manufacturer repair guidance)
- **Crash test databases** — Engineering signal data (CTD API)

---

## Research Paper Ideas (ranked by novelty × feasibility)

### 1. "Mining Consumer Complaints for Early Defect Detection: A Large Language Model Approach"

**Core idea:** Use LLMs to extract structured defect signals from NHTSA's 30+ years of free-text consumer complaints (1995+), then analyze temporal patterns to identify defects that were *not* caught by NHTSA's existing recall process.

**Data sources:**
- Complaints DB (free-text narratives, 1995-2026, ~400MB of text)
- ODI Recalls (to compare detected vs. recalled defects)
- vPIC (vehicle specs to correlate defect patterns with model/year)

**Methodology:**
1. **Structured extraction** — LLMs extract: defect category, severity, symptoms, affected subsystem, mileage range, recurrence pattern
2. **Temporal clustering** — group complaint narratives by semantic similarity using embeddings; detect spikes in specific defect clusters
3. **Gap analysis** — compare LLM-detected defect clusters against NHTSA's formal recall timeline; identify "missed" defects (clusters that spiked but never triggered a recall)
4. **Validation** — cross-reference with TSBS (Technical Service Bulletins) and warranty data proxies

**Novelty:**
- First systematic LLM analysis of all 30+ years of NHTSA complaints
- Identifies *recalls that should have happened* (false negatives in NHTSA's own process)
- Quantifies detection lag: time between first complaint cluster and formal recall

**Contributions:**
- Framework for automated defect signal detection from unstructured consumer reports
- Empirical analysis of NHTSA recall process gaps
- Open dataset of LLM-extracted defect annotations from NHTSA complaints

**Feasibility:** High — all data is public, no API rate limits on bulk downloads, LLM extraction is straightforward

---

### 2. "ADAS Technology Adoption vs. Real-World Safety: A Multi-Source Analysis Using NHTSA Data and LLMs"

**Core idea:** Correlate the presence of specific ADAS features (from vPIC) with actual crash outcomes (from FARS/NCAP) to quantify the real-world safety impact of individual ADAS technologies.

**Data sources:**
- vPIC — ADAS/ADS tech presence by make/model/year (ACC, FCW, LDW, LKA, AEB, etc., for MY 2017+)
- NCAP — crash avoidance test scores for the same vehicles
- FARS — fatal crash data (person, vehicle, safety equipment, weather, distraction datasets)
- Complaints — consumer-reported ADAS failures

**Methodology:**
1. Build a vehicle-level dataset: make/model/year × ADAS features (binary) × NCAP scores × crash involvement
2. Use LLMs to extract ADAS-related complaints and classify failure modes (false positive, false negative, driver confusion, system override)
3. Statistical analysis: control for vehicle weight, age, driver demographics to estimate causal impact of each ADAS feature on crash fatality rate
4. LLM-based narrative analysis of FARS "distracted driving" and "safety equipment" fields to detect ADAS-related crash patterns

**Novelty:**
- First study combining manufacturer-reported ADAS presence (vPIC) with actual fatal crash outcomes (FARS) at scale
- LLM analysis of NHTSA complaint narratives to identify ADAS failure modes
- Addresses the gap between NCAP lab test results and real-world performance

**Contributions:**
- Empirical quantification of real-world ADAS safety impact
- Taxonomy of ADAS failure modes from consumer reports
- Evidence for/against current NCAP testing methodology

**Feasibility:** Medium — requires joining multiple datasets, statistical controls needed

---

### 3. "Automated Analysis of Autonomous Vehicle Incident Reports: LLM-Driven Classification of SGO 2021-01 ADS Incidents"

**Core idea:** Apply LLMs to systematically classify and analyze NHTSA's Standing General Order 2021-01 ADS incident reports, extracting structured causal factors, contributing conditions, and safety implications from unstructured narrative descriptions.

**Data sources:**
- SGO 2021-01 incident reports (monthly CSV releases with free-text narratives)
- vPIC (vehicle specs for involved vehicles)
- FARS (cross-reference with fatal crashes)

**Methodology:**
1. LLM classification of incident types: sensor failure, planning error, prediction error, system takeover failure, infrastructure-related, weather-related, other
2. Temporal trend analysis: track incident types over time (2022-present)
3. Cross-reference ADS incidents with ADAS incidents to compare failure mode distributions
4. LLM-based extraction of contributing factors (road conditions, weather, human intervention, system version)

**Novelty:**
- First systematic analysis of all NHTSA ADS incident reports
- Direct comparison of ADAS vs. ADS failure modes
- Temporal tracking of ADS safety as the technology matures

**Contributions:**
- Structured taxonomy of ADS incident types from unstructured reports
- Evidence-based assessment of ADS safety trajectory
- Framework for ongoing LLM monitoring of ADS incident data

**Feasibility:** High — data is small (CSV), narratives are well-structured, monthly updates provide natural evaluation timeline

---

### 4. "From Narrative to Signal: LLM-Based Recall Prediction Using NHTSA Consumer Complaints"

**Core idea:** Build a predictive model that uses LLM-extracted features from consumer complaints to predict which vehicles will receive safety recalls, with lead time analysis.

**Data sources:**
- Complaints DB (free-text narratives, structured fields)
- ODI Recalls (recall announcements with dates)
- vPIC (vehicle specifications)

**Methodology:**
1. LLM feature extraction from complaints: defect severity (1-5), safety criticality, recurrence pattern, affected population estimate
2. Time-series analysis: build complaint intensity features per vehicle model over time
3. Train a recall prediction model: predict whether a vehicle model will receive a recall in the next 6-12 months
4. Analyze which complaint features are most predictive of recalls
5. Identify "silent" complaint clusters — high-intensity complaints that never led to recalls

**Novelty:**
- First predictive model for safety recalls using NHTSA complaint data
- LLM-based severity/criticality extraction from unstructured complaints
- Quantifies the "recall detection gap" — complaints that should have triggered recalls

**Contributions:**
- Predictive framework for consumer safety advocacy
- Evidence-based analysis of NHTSA recall decision-making
- Open tool for real-time recall risk monitoring

**Feasibility:** Medium — requires careful temporal splitting to avoid data leakage, needs domain expertise for validation

---

### 5. "Crash Reconstruction Assistant: LLM-Powered Analysis of NHTSA Crash Data for Safety Research"

**Core idea:** Build an LLM-powered research assistant that combines multiple NHTSA crash datasets (FARS, NCAP, vPIC, complaints) to generate comprehensive crash narratives and safety recommendations.

**Data sources:**
- FARS (fatal crash data, multiple datasets)
- NCAP (crash test ratings for involved vehicles)
- vPIC (vehicle specs for involved vehicles)
- Complaints (prior complaints for involved vehicles)

**Methodology:**
1. LLM-based crash narrative generation: combine structured crash data with vehicle specs and prior complaints to generate detailed crash analysis reports
2. LLM-based safety recommendation extraction: analyze crash patterns and generate targeted safety recommendations
3. Benchmark LLM-generated analyses against NHTSA's own crash investigation reports
4. Evaluate LLM accuracy in identifying contributing factors, crash dynamics, and preventability

**Novelty:**
- First multi-source NHTSA crash analysis system powered by LLMs
- Combines engineering data (NCAP crash tests) with real-world outcomes (FARS)
- Generates actionable safety recommendations from raw crash data

**Contributions:**
- Open-source crash analysis pipeline
- Benchmark for LLM-based safety analysis
- Tool for accelerated safety research

**Feasibility:** Medium-High — requires careful handling of sensitive crash data, but all sources are public

---

## Recommended Paper (Best Balance)

**Paper 1: "Mining Consumer Complaints for Early Defect Detection"**

**Why this one:**
- Highest novelty: 30+ years of complaints have never been systematically analyzed with LLMs
- Highest feasibility: bulk downloads available, no auth, no rate limits
- Strong impact: identifies systemic gaps in NHTSA's recall process
- Clean evaluation: compare LLM detections against known recalls (ground truth available)
- Extensible: monthly complaint updates provide ongoing evaluation opportunity
- Fits your background: ADAS/vehicle safety domain expertise is directly relevant

**Paper structure:**
1. Introduction: NHTSA's complaint system, limitations of current analysis methods
2. Related work: defect detection, text mining in safety domains, LLMs for regulatory analysis
3. Data: NHTSA complaints, recalls, vPIC — dataset statistics, preprocessing
4. Method: LLM extraction pipeline, temporal clustering, gap analysis framework
5. Results: defect detection accuracy, temporal patterns, identified recall gaps
6. Discussion: implications for NHTSA, limitations, future work
7. Conclusion

**Timeline:** ~3-4 months for data collection, LLM processing, analysis, and writing
