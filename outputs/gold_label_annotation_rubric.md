# Gold-Label Annotation Rubric for 2018 NHTSA Complaint Sample

Date: 2026-09-30  
Applies to: `data/gold_sample_2018.csv`  
Purpose: create consistent human labels for evaluating local-LM-Studio extraction of vehicle defect events.

## 1. Annotation goal

Label what the complaint narrative supports, not what may be true externally. Use the NHTSA metadata fields as context, but prefer the consumer narrative when assigning failure mode, severity, and safety outcome.

Each row is one complaint/component record. If the same `ODI Number` appears in multiple rows, annotate each row independently unless the narrative clearly applies to all listed components.

## 2. Fields to complete

### `gold_component`

Use the most specific vehicle system/component implicated by the complaint.

Recommended controlled values:

- `engine`
- `power_train_transmission`
- `service_brakes`
- `steering`
- `suspension`
- `air_bags`
- `seat_belts`
- `electrical_system`
- `fuel_propulsion_system`
- `vehicle_speed_control`
- `wheels_tires`
- `structure_body`
- `visibility_wipers_glass`
- `exterior_lighting`
- `seats`
- `adas_electronics_automation`
- `unknown_or_other`
- `not_a_defect_complaint`

Rules:
- If multiple components are plausibly involved, choose the component most directly tied to the hazardous event.
- If the narrative only says “car failed” or is too vague, use `unknown_or_other`.
- If the complaint is mostly about dealer service, warranty, price, inconvenience, or recall notification without a defect event, use `not_a_defect_complaint`.

### `gold_failure_mode`

Short free-text phrase describing the concrete failure.

Examples:
- `engine stalls while driving`
- `brake pedal goes to floor`
- `airbag warning light on / nondeployment concern`
- `loss of power steering assist`
- `unintended acceleration`
- `fuel leak odor`
- `tire tread separation`

Rules:
- Use 3–10 words when possible.
- Do not copy the full narrative.
- Do not infer a root cause unless explicitly stated. Prefer symptom/event language over engineering diagnosis.

### `gold_severity_1_5`

Use this ordinal scale:

| Score | Meaning | Examples |
|---:|---|---|
| 1 | Low safety relevance or nuisance defect | cosmetic issue, infotainment problem, non-safety warning with no drivability issue |
| 2 | Potential safety relevance but low immediacy | intermittent warning light, minor visibility issue, noncritical malfunction |
| 3 | Clear safety-relevant malfunction without crash/injury/fire | stalling, brake degradation, steering assist loss, airbag warning |
| 4 | High-risk event or near miss | loss of braking/steering at speed, unintended acceleration, stall in traffic, smoke/fire risk without injury |
| 5 | Crash, fire, injury, death, or severe imminent hazard | reported crash, fire, injury, fatality, wheel separation at speed |

Rules:
- Severity should reflect the event described, not anger/frustration in the narrative.
- Use NHTSA `CRASH`, `FIRE`, `INJURED`, `DEATHS` fields to support severity 5, but check the narrative for consistency.
- If uncertain between two levels, choose the lower level and explain uncertainty in `gold_notes`.

### `gold_safety_outcome`

Use one controlled value:

- `none_reported`
- `warning_light_or_message`
- `loss_of_control_risk`
- `stall_or_loss_of_power`
- `braking_impairment`
- `steering_impairment`
- `visibility_impairment`
- `airbag_seatbelt_nonperformance_risk`
- `fire_smoke_or_fuel_leak`
- `crash`
- `injury`
- `death`
- `other_safety_outcome`
- `unclear`

Rules:
- Choose the most severe outcome explicitly supported.
- If both crash and injury are present, choose `injury`; if death is present, choose `death`.
- If the narrative describes risk but no realized outcome, choose the risk category, not `crash` or `injury`.

### `gold_crash_fire_injury_death`

Encode explicit severe outcome flags as a compact semicolon-separated list.

Allowed tokens:

- `crash`
- `fire`
- `injury`
- `death`
- `none`
- `unclear`

Examples:
- `none`
- `crash`
- `crash;injury`
- `fire`

Rules:
- Use `none` when the complaint does not report crash/fire/injury/death.
- Use `unclear` when metadata and narrative conflict or the narrative is ambiguous.

### `gold_notes`

Optional but encouraged. Use for:

- ambiguity,
- multiple components,
- metadata/narrative disagreement,
- suspected non-defect complaint,
- root cause explicitly stated by dealer/mechanic,
- why a severity score was chosen.

Keep notes brief: one sentence is usually enough.

### `reviewer_id`

Use a stable reviewer code, e.g. `gp`, `r1`, `r2`.

## 3. General decision rules

1. **Do not over-infer.** If the complaint says the engine stalled, label `engine stalls while driving`; do not infer fuel pump, ignition, or software unless stated.
2. **Prefer event over cause.** The extraction task should learn defect-event structure before root-cause diagnosis.
3. **Keep ambiguous cases visible.** Use `unknown_or_other`, `unclear`, and `gold_notes` rather than forcing precision.
4. **Separate severity from frequency.** Repeated trips to the dealer may support recurrence, but severity depends on safety consequence.
5. **Use metadata cautiously.** NHTSA `Component` is useful context but may be broad or noisy.
6. **Do not label recalls.** The row is a complaint label, not a determination that a recall exists or should exist.

## 4. Minimal quality check before using labels

Before evaluating the LLM, run these checks on the completed CSV:

- no blank `gold_component`, `gold_failure_mode`, `gold_severity_1_5`, `gold_safety_outcome`, or `reviewer_id`;
- all `gold_severity_1_5` values are integers 1–5;
- controlled fields contain only allowed values;
- at least 20 rows with `safety_flag` stratum were reviewed carefully for metadata/narrative consistency;
- optionally double-label 50 rows and measure inter-annotator agreement before finalizing the schema.

## 5. Recommended pilot acceptance criteria

Do not move to large-scale LLM extraction until:

- at least 300 rows are manually labeled;
- schema disagreements are resolved and documented;
- at least 50 high-safety rows are included;
- a small adjudication pass has corrected obvious inconsistent labels;
- evaluation code reports per-field accuracy/precision/recall rather than only an overall score.
