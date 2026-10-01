"""
Defect annotation schema — formal definitions from the paper.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Defect Categories (24 categories, 5 groups) ──────────────────────────

class DefectGroup(str, Enum):
    POWERTRAIN = "powertrain"
    ELECTRICAL = "electrical_electronic"
    BRAKES_STEERING_SUSPENSION = "brakes_steering_suspension"
    BODY_STRUCTURAL = "body_structural"
    ADVANCED_SYSTEMS = "advanced_systems"


class DefectCategory(str, Enum):
    # Powertrain
    ENGINE_STALLING = "engine_stalling_shutdown"
    TRANSMISSION = "transmission_failure"
    ENGINE_PERFORMANCE = "engine_performance_degradation"
    FUEL_SYSTEM = "fuel_system_defect"
    EXHAUST = "exhaust_emissions_defect"
    # Electrical/Electronic
    ELECTRICAL_SYSTEM = "electrical_system_failure"
    BATTERY = "battery_charging_defect"
    INFOTAINMENT = "infotainment_navigation_failure"
    KEYLESS_IGNITION = "keyless_ignition_defect"
    WARNING_LIGHT = "warning_light_malfunction"
    # Brakes/Steering/Suspension
    BRAKES = "brake_system_failure"
    STEERING = "steering_defect"
    SUSPENSION = "suspension_defect"
    TIRE_WHEEL = "tire_wheel_defect"
    # Body/Structural
    BODY_TRIM = "body_trim_defect"
    STRUCTURAL = "structural_corrosion_defect"
    GLASS = "glass_window_defect"
    SEAT_RESTRAINT = "seat_restraint_defect"
    # Advanced Systems
    ADAS = "adas_sensor_failure"
    AUTONOMOUS = "autonomous_driving_defect"
    HVAC = "hvac_defect"
    OTHER = "other"

    @property
    def group(self) -> DefectGroup:
        mapping = {
            "engine_stalling_shutdown": DefectGroup.POWERTRAIN,
            "transmission_failure": DefectGroup.POWERTRAIN,
            "engine_performance_degradation": DefectGroup.POWERTRAIN,
            "fuel_system_defect": DefectGroup.POWERTRAIN,
            "exhaust_emissions_defect": DefectGroup.POWERTRAIN,
            "electrical_system_failure": DefectGroup.ELECTRICAL,
            "battery_charging_defect": DefectGroup.ELECTRICAL,
            "infotainment_navigation_failure": DefectGroup.ELECTRICAL,
            "keyless_ignition_defect": DefectGroup.ELECTRICAL,
            "warning_light_malfunction": DefectGroup.ELECTRICAL,
            "brake_system_failure": DefectGroup.BRAKES_STEERING_SUSPENSION,
            "steering_defect": DefectGroup.BRAKES_STEERING_SUSPENSION,
            "suspension_defect": DefectGroup.BRAKES_STEERING_SUSPENSION,
            "tire_wheel_defect": DefectGroup.BRAKES_STEERING_SUSPENSION,
            "body_trim_defect": DefectGroup.BODY_STRUCTURAL,
            "structural_corrosion_defect": DefectGroup.BODY_STRUCTURAL,
            "glass_window_defect": DefectGroup.BODY_STRUCTURAL,
            "seat_restraint_defect": DefectGroup.BODY_STRUCTURAL,
            "adas_sensor_failure": DefectGroup.ADVANCED_SYSTEMS,
            "autonomous_driving_defect": DefectGroup.ADVANCED_SYSTEMS,
            "hvac_defect": DefectGroup.ADVANCED_SYSTEMS,
            "other": DefectGroup.ADVANCED_SYSTEMS,
        }
        return mapping[self.value]

    @property
    def short_code(self) -> str:
        mapping = {
            "engine_stalling_shutdown": "E1",
            "transmission_failure": "E2",
            "engine_performance_degradation": "E3",
            "fuel_system_defect": "E4",
            "exhaust_emissions_defect": "E5",
            "electrical_system_failure": "EL1",
            "battery_charging_defect": "EL2",
            "infotainment_navigation_failure": "EL3",
            "keyless_ignition_defect": "EL4",
            "warning_light_malfunction": "EL5",
            "brake_system_failure": "B1",
            "steering_defect": "B2",
            "suspension_defect": "B3",
            "tire_wheel_defect": "B4",
            "body_trim_defect": "BD1",
            "structural_corrosion_defect": "BD2",
            "glass_window_defect": "BD3",
            "seat_restraint_defect": "BD4",
            "adas_sensor_failure": "AD1",
            "autonomous_driving_defect": "AD2",
            "hvac_defect": "AD3",
            "other": "OD",
        }
        return mapping[self.value]

    @property
    def label(self) -> str:
        mapping = {
            "engine_stalling_shutdown": "Engine stalling/shutdown",
            "transmission_failure": "Transmission failure",
            "engine_performance_degradation": "Engine performance degradation",
            "fuel_system_defect": "Fuel system defect",
            "exhaust_emissions_defect": "Exhaust/emissions defect",
            "electrical_system_failure": "Electrical system failure",
            "battery_charging_defect": "Battery/charging defect",
            "infotainment_navigation_failure": "Infotainment/navigation failure",
            "keyless_ignition_defect": "Keyless ignition defect",
            "warning_light_malfunction": "Warning light malfunction",
            "brake_system_failure": "Brake system failure",
            "steering_defect": "Steering defect",
            "suspension_defect": "Suspension defect",
            "tire_wheel_defect": "Tire/wheel defect",
            "body_trim_defect": "Body/trim defect",
            "structural_corrosion_defect": "Structural/corrosion defect",
            "glass_window_defect": "Glass/window defect",
            "seat_restraint_defect": "Seat/restraint defect",
            "adas_sensor_failure": "ADAS/sensor failure",
            "autonomous_driving_defect": "Autonomous driving defect",
            "hvac_defect": "HVAC defect",
            "other": "Other",
        }
        return mapping[self.value]


# ── Subsystem Categories (15 categories) ─────────────────────────────────

class Subsystem(str, Enum):
    ENGINE = "engine"
    TRANSMISSION = "transmission"
    BRAKES = "brakes"
    STEERING = "steering"
    SUSPENSION = "suspension"
    ELECTRICAL = "electrical"
    FUEL_SYSTEM = "fuel_system"
    EXHAUST = "exhaust"
    BODY = "body"
    SEATS = "seats"
    GLASS = "glass"
    TIRES = "tires"
    ADAS = "adas_sensors"
    INFOTAINMENT = "infotainment"
    OTHER = "other"

    @property
    def label(self) -> str:
        mapping = {
            "engine": "Engine",
            "transmission": "Transmission",
            "brakes": "Brakes",
            "steering": "Steering",
            "suspension": "Suspension",
            "electrical": "Electrical",
            "fuel_system": "Fuel System",
            "exhaust": "Exhaust",
            "body": "Body",
            "seats": "Seats",
            "glass": "Glass",
            "tires": "Tires",
            "adas_sensors": "ADAS/Sensors",
            "infotainment": "Infotainment",
            "other": "Other",
        }
        return mapping[self.value]


# ── 28 Symptom Patterns ──────────────────────────────────────────────────

SYMPTOM_NAMES: list[str] = [
    "intermittent_failure",           # 0
    "progressive_worsening",          # 1
    "specific_condition_failure",     # 2: cold/heat/rain/highway
    "failure_after_repair",           # 3
    "multiple_symptoms",              # 4
    "noise_vibration",                # 5
    "warning_light_alert",            # 6
    "vehicle_immobilized",            # 7
    "loss_of_power",                  # 8
    "loss_of_braking",                # 9
    "loss_of_steering",               # 10
    "fire_smoke",                     # 11
    "fluid_leak",                     # 12
    "electrical_short",               # 13
    "software_glitch",                # 14
    "part_breakage",                  # 15
    "corrosion_rust",                 # 16
    "recall_related",                 # 17
    "dealer_visit",                   # 18
    "tow_truck",                      # 19
    "injury_reported",                # 20
    "property_damage",                # 21
    "safety_risk_stated",             # 22
    "manufacturer_contacted",         # 23
    "warranty_claim",                 # 24
    "used_vehicle",                   # 25
    "high_mileage",                   # 26: >100k
    "low_mileage",                    # 27: <10k
]


# ── Severity Scale ───────────────────────────────────────────────────────

class Severity(int, Enum):
    COSMETIC = 1      # Cosmetic/minor
    INCONVENIENCE = 2 # Inconvenience
    MODERATE = 3      # Moderate safety concern
    HIGH = 4          # High safety risk
    CATASTROPHIC = 5  # Crash/catastrophic

    @property
    def label(self) -> str:
        return {
            1: "Cosmetic/minor",
            2: "Inconvenience",
            3: "Moderate safety concern",
            4: "High safety risk",
            5: "Catastrophic/crash",
        }[self.value]


# ── Defect Annotation ────────────────────────────────────────────────────

class DefectAnnotation(BaseModel):
    """Structured defect annotation extracted from a complaint narrative.

    Corresponds to Definition 1 in the paper:
    A = (c, s, sub, sym, rec, temp, conf)
    """

    complaint_id: str
    make: str
    model: str
    model_year: int
    date_received: str  # ISO format
    narrative: str

    # Extracted fields
    defect_category: DefectCategory
    severity: Severity
    subsystem: Subsystem
    symptoms: list[str] = Field(default_factory=list, description="Symptom pattern names")
    symptom_vector: list[int] = Field(default_factory=lambda: [0] * 28, description="28-element binary vector")
    recurrence: bool = False
    time_to_failure_months: float | None = None
    confidence: float = 1.0

    # Metadata
    extraction_model: str = "qwen-3.6-35b-a3b"
    extraction_timestamp: str = ""

    @field_validator("symptom_vector")
    @classmethod
    def validate_symptom_vector(cls, v: list[int]) -> list[int]:
        if len(v) != 28:
            raise ValueError(f"symptom_vector must have 28 elements, got {len(v)}")
        if any(x not in (0, 1) for x in v):
            raise ValueError("symptom_vector must contain only 0 or 1")
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: Severity) -> Severity:
        if v not in (Severity.COSMETIC, Severity.INCONVENIENCE, Severity.MODERATE,
                      Severity.HIGH, Severity.CATASTROPHIC):
            raise ValueError(f"severity must be 1-5, got {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON/Parquet output."""
        return {
            "complaint_id": self.complaint_id,
            "make": self.make,
            "model": self.model,
            "model_year": self.model_year,
            "date_received": self.date_received,
            "narrative": self.narrative,
            "defect_category": self.defect_category.value,
            "defect_category_label": self.defect_category.label,
            "defect_category_code": self.defect_category.short_code,
            "defect_group": self.defect_category.group.value,
            "severity": self.severity.value,
            "severity_label": self.severity.label,
            "subsystem": self.subsystem.value,
            "subsystem_label": self.subsystem.label,
            "symptoms": self.symptoms,
            "symptom_vector": self.symptom_vector,
            "recurrence": self.recurrence,
            "time_to_failure_months": self.time_to_failure_months,
            "confidence": self.confidence,
            "extraction_model": self.extraction_model,
            "extraction_timestamp": self.extraction_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DefectAnnotation":
        """Deserialize from dict."""
        return cls(
            complaint_id=data["complaint_id"],
            make=data["make"],
            model=data["model"],
            model_year=data["model_year"],
            date_received=data["date_received"],
            narrative=data["narrative"],
            defect_category=DefectCategory(data["defect_category"]),
            severity=Severity(data["severity"]),
            subsystem=Subsystem(data["subsystem"]),
            symptoms=data.get("symptoms", []),
            symptom_vector=data.get("symptom_vector", [0] * 28),
            recurrence=data.get("recurrence", False),
            time_to_failure_months=data.get("time_to_failure_months"),
            confidence=data.get("confidence", 1.0),
            extraction_model=data.get("extraction_model", "unknown"),
            extraction_timestamp=data.get("extraction_timestamp", ""),
        )

    @property
    def is_safety_relevant(self) -> bool:
        """Check if annotation represents a safety-relevant defect (severity >= 3)."""
        return self.severity.value >= 3

    @property
    def is_high_severity(self) -> bool:
        """Check if annotation represents a high-severity defect (severity >= 4)."""
        return self.severity.value >= 4


# ── Prompt Template for LLM Extraction ───────────────────────────────────

SYSTEM_PROMPT = """You are an automotive safety analyst with expertise in vehicle defect identification.
Your task is to analyze consumer complaint narratives and extract structured defect annotations.

OUTPUT ONLY valid JSON matching this schema:
{
  "defect_category": "<one of the 24 categories listed below>",
  "severity": <integer 1-5>,
  "subsystem": "<one of the 15 subsystems listed below>",
  "symptoms": ["<symptom pattern names from the list below>"],
  "recurrence": <true/false>,
  "time_to_failure_months": <number or null>,
  "confidence": <float 0.0-1.0>
}

DEFE CATEGORIES (choose exactly one):
Powertrain: engine_stalling_shutdown, transmission_failure, engine_performance_degradation, fuel_system_defect, exhaust_emissions_defect
Electrical/Electronic: electrical_system_failure, battery_charging_defect, infotainment_navigation_failure, keyless_ignition_defect, warning_light_malfunction
Brakes/Steering/Suspension: brake_system_failure, steering_defect, suspension_defect, tire_wheel_defect
Body/Structural: body_trim_defect, structural_corrosion_defect, glass_window_defect, seat_restraint_defect
Advanced Systems: adas_sensor_failure, autonomous_driving_defect, hvac_defect, other

SUBSYSTEMS (choose exactly one):
engine, transmission, brakes, steering, suspension, electrical, fuel_system, exhaust, body, seats, glass, tires, adas_sensors, infotainment, other

SYMPTOM PATTERNS (list all that apply):
intermittent_failure, progressive_worsening, specific_condition_failure, failure_after_repair,
multiple_symptoms, noise_vibration, warning_light_alert, vehicle_immobilized, loss_of_power,
loss_of_braking, loss_of_steering, fire_smoke, fluid_leak, electrical_short, software_glitch,
part_breakage, corrosion_rust, recall_related, dealer_visit, tow_truck, injury_reported,
property_damage, safety_risk_stated, manufacturer_contacted, warranty_claim, used_vehicle,
high_mileage, low_mileage

SEVERITY SCALE:
1 = Cosmetic/minor
2 = Inconvenience
3 = Moderate safety concern
4 = High safety risk
5 = Catastrophic/crash

RULES:
1. Base severity on the SAFETY IMPLICATION described in the narrative, not just the component.
2. If the narrative mentions injury, crash, or loss of vehicle control, severity is at least 4.
3. If the narrative mentions fire, smoke, or fluid leak near hot components, severity is at least 4.
4. For ADAS/autonomous systems, treat unexpected behavior as at least severity 3.
5. Confidence should reflect narrative quality: short/unclear narratives get lower confidence.
6. time_to_failure_months: estimate months from purchase to first occurrence. Use null if unknown.
7. recurrence=true if the narrative mentions the problem happening multiple times, intermittently, or after repair.
8. Do NOT add fields not in the schema. Do NOT include any text outside the JSON.
"""

FEW_SHOT_EXAMPLES = """

EXAMPLE 1:
Narrative: "My 2019 Honda Accord's brake pedal feels spongy and requires increasingly hard pressure to stop. Started 6 months ago and has gotten progressively worse. Dealer replaced brake fluid twice but problem persists. Very concerning as this affects my ability to stop safely."
Output: {
  "defect_category": "brake_system_failure",
  "severity": 4,
  "subsystem": "brakes",
  "symptoms": ["progressive_worsening", "dealer_visit", "safety_risk_stated"],
  "recurrence": true,
  "time_to_failure_months": 6,
  "confidence": 0.95
}

EXAMPLE 2:
Narrative: "2021 Tesla Model 3 autopilot disengages randomly on highway at 65 mph. Happens maybe once a week. No warning before disengagement. Scary because car is changing lanes without me. Dealer says 'normal behavior' but this feels unsafe."
Output: {
  "defect_category": "adas_sensor_failure",
  "severity": 4,
  "subsystem": "adas_sensors",
  "symptoms": ["intermittent_failure", "safety_risk_stated", "vehicle_immobilized"],
  "recurrence": true,
  "time_to_failure_months": null,
  "confidence": 0.88
}

EXAMPLE 3:
Narrative: "Paint is peeling off the hood of my 2020 Ford F-150 after just 18 months. Looks terrible but doesn't affect driving. Dealer offered to touch it up."
Output: {
  "defect_category": "body_trim_defect",
  "severity": 2,
  "subsystem": "body",
  "symptoms": ["progressive_worsening"],
  "recurrence": false,
  "time_to_failure_months": 18,
  "confidence": 0.90
}

EXAMPLE 4:
Narrative: "Engine stalls at idle every time I stop at a red light. Makes me nervous that I'll stall in traffic. Had it towed once when it wouldn't restart. Mechanic couldn't find anything wrong."
Output: {
  "defect_category": "engine_stalling_shutdown",
  "severity": 4,
  "subsystem": "engine",
  "symptoms": ["intermittent_failure", "loss_of_power", "tow_truck", "safety_risk_stated"],
  "recurrence": true,
  "time_to_failure_months": null,
  "confidence": 0.85
}
"""


def build_extraction_prompt(narrative: str) -> str:
    """Build the full prompt for a single complaint narrative."""
    return f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLES}

CURRENT COMPLAINT:
Make: N/A  Model: N/A  Year: N/A
Narrative: {narrative}

Output:"""


def build_batch_prompt(narratives: list[str]) -> list[str]:
    """Build prompts for a batch of narratives."""
    return [build_extraction_prompt(n) for n in narratives]
