"""
ingestion/ontology.py
=====================
Lightweight ontology for thermal and electrical energy utilities.

This module provides:
1. Tag dictionaries mapping canonical concepts to keyword synonyms
2. Functions to detect relevant tags in text
3. Equipment and concept classification for metadata enrichment

This ontology is used during ingestion to tag every chunk with
relevant domain concepts, enabling metadata-filtered retrieval later.

Domain coverage:
- THERMAL: combustion, boilers, steam, furnaces, cogeneration, WHR, etc.
- ELECTRICAL: tariffs, motors, power factor, drives, HVAC, lighting, etc.
"""

from __future__ import annotations
import re
from typing import NamedTuple


# ============================================================
# ONTOLOGY DEFINITIONS
# ============================================================


# Each entry: canonical_tag -> list of keyword patterns to detect it
# Patterns are matched case-insensitively as whole words / substrings

THERMAL_EQUIPMENT_TAGS: dict[str, list[str]] = {
    "boiler": ["boiler", "steam generator", "fire tube", "water tube"],
    "furnace": ["furnace", "kiln", "oven", "reheating furnace", "annealing"],
    "steam_system": ["steam", "steam trap", "condensate", "flash steam", "steam header"],
    "heat_exchanger": ["heat exchanger", "recuperator", "economiser", "economizer", "preheater"],
    "insulation_refractory": ["insulation", "refractory", "lagging", "cladding", "thermal block"],
    "combustion_system": ["burner", "combustion", "flame", "ignition", "excess air", "stoichiometric"],
    "cogeneration": ["cogeneration", "cogen", "combined heat and power", "chp", "topping cycle", "bottoming cycle"],
    "waste_heat_recovery": ["waste heat", "heat recovery", "flue gas", "exhaust heat", "recuperation", "regeneration"],
    "fbc_boiler": ["fbc", "fluidised bed", "fluidized bed", "cfbc", "bubbling bed"],
    "fuel_system": ["fuel", "coal", "oil", "natural gas", "lng", "lpg", "calorific value", "gvc", "nvc"],
    "steam_trap": ["steam trap", "thermostatic", "mechanical trap", "thermodynamic trap"],
    "cooling_tower_thermal": ["cooling tower"],
    "dryer": ["dryer", "drying", "moisture removal"],
    "compressor_thermal": ["compressed air", "air compressor"],
}

THERMAL_CONCEPT_TAGS: dict[str, list[str]] = {
    "combustion": ["combustion", "burning", "stoichiometry", "air-fuel ratio", "excess air", "co2", "flue gas analysis"],
    "heat_transfer": ["heat transfer", "conduction", "convection", "radiation", "thermal conductivity", "u-value", "lmtd"],
    "energy_audit": ["energy audit", "energy assessment", "energy survey"],
    "efficiency": ["efficiency", "thermal efficiency", "boiler efficiency", "furnace efficiency", "indirect method", "direct method"],
    "steam_quality": ["steam quality", "dryness fraction", "wetness", "superheated steam", "saturated steam"],
    "pinch_analysis": ["pinch", "pinch analysis", "pinch point", "heat integration", "minimum utility"],
    "insulation_loss": ["heat loss", "insulation loss", "bare surface loss", "lagging"],
    "blowdown": ["blowdown", "dissolved solids", "tds", "total dissolved solids"],
    "enthalpy": ["enthalpy", "latent heat", "sensible heat", "specific heat"],
    "temperature": ["temperature", "flue gas temperature", "stack temperature", "exit temperature"],
    "pressure": ["pressure", "steam pressure", "operating pressure", "header pressure"],
    "condensate_recovery": ["condensate recovery", "condensate return", "flash vessel"],
    "load_management_thermal": ["load management", "peak load", "load scheduling"],
    "carbon_emission": ["co2", "carbon", "emission", "greenhouse"],
}

ELECTRICAL_EQUIPMENT_TAGS: dict[str, list[str]] = {
    "motor": ["motor", "induction motor", "electric motor", "ac motor"],
    "vfd": ["vfd", "variable frequency drive", "variable speed drive", "vsd", "inverter drive", "speed controller"],
    "transformer": ["transformer", "distribution transformer", "step-down", "step-up", "no-load loss", "core loss"],
    "capacitor_bank": ["capacitor", "capacitor bank", "shunt capacitor", "power factor correction"],
    "compressed_air_system": ["compressed air", "air compressor", "air receiver", "pneumatic", "cfm", "scfm"],
    "pump": ["pump", "centrifugal pump", "reciprocating pump", "submersible pump"],
    "fan_blower": ["fan", "blower", "axial fan", "centrifugal fan", "duct fan", "induced draft", "forced draft"],
    "hvac": ["hvac", "air conditioning", "chiller", "ahu", "air handling unit", "vav", "fcu"],
    "refrigeration": ["refrigeration", "refrigerant", "cop", "coefficient of performance", "chiller", "evaporator", "condenser"],
    "cooling_tower_elec": ["cooling tower", "ct", "cooling water"],
    "lighting": ["lighting", "luminaire", "led", "fluorescent", "lux", "lamp", "luminous efficacy"],
    "dg_set": ["dg set", "diesel generator", "genset", "standby generator"],
    "ups": ["ups", "uninterruptible power supply"],
    "switchgear": ["switchgear", "panel board", "distribution board", "mccb", "acb"],
}

ELECTRICAL_CONCEPT_TAGS: dict[str, list[str]] = {
    "tariff": ["tariff", "electricity tariff", "tod", "time of day", "tou", "billing", "unit cost", "kvah"],
    "power_factor": ["power factor", "pf", "reactive power", "kvar", "apparent power", "kva", "unity pf"],
    "maximum_demand": ["maximum demand", "md", "demand charge", "peak demand", "rmd"],
    "load_factor": ["load factor", "load curve", "diversity factor", "utilisation factor"],
    "harmonics": ["harmonic", "thd", "total harmonic distortion", "harmonic filter"],
    "energy_management": ["energy management", "energy monitoring", "energy accounting", "ems"],
    "energy_audit_elec": ["energy audit", "energy survey", "power audit"],
    "loss_reduction": ["loss reduction", "copper loss", "iron loss", "i2r loss", "technical losses"],
    "automation": ["automation", "plc", "scada", "bms", "building management"],
    "ecbc": ["ecbc", "energy conservation building code", "building energy"],
    "sub_metering": ["sub-metering", "sub metering", "energy meter", "kwh meter"],
    "power_quality": ["power quality", "voltage sag", "voltage swell", "flicker", "interruption"],
    "contract_demand": ["contract demand", "sanctioned demand", "connected load"],
    "reactive_compensation": ["reactive compensation", "capacitor", "synchronous condenser", "kvar"],
    "motor_efficiency": ["motor efficiency", "ie1", "ie2", "ie3", "efficiency class", "nameplate", "rewinding"],
    "vfd_savings": ["vfd savings", "speed control", "affinity laws", "fan law", "pump law"],
}


# ============================================================
# COMBINED TAG LOOKUP
# ============================================================


ALL_THERMAL_TAGS: dict[str, list[str]] = {
    **THERMAL_EQUIPMENT_TAGS,
    **THERMAL_CONCEPT_TAGS,
}

ALL_ELECTRICAL_TAGS: dict[str, list[str]] = {
    **ELECTRICAL_EQUIPMENT_TAGS,
    **ELECTRICAL_CONCEPT_TAGS,
}


class TagResult(NamedTuple):
    equipment_tags: list[str]
    concept_tags: list[str]


def detect_tags(text: str, domain: str) -> TagResult:
    """
    Detect relevant tags in a chunk of text by matching against
    the ontology keyword lists.

    Args:
        text: The chunk text to analyze.
        domain: "thermal" or "electrical"

    Returns:
        TagResult with detected equipment_tags and concept_tags.
    """
    text_lower = text.lower()
    equipment_tags: list[str] = []
    concept_tags: list[str] = []

    if domain == "thermal":
        eq_ontology = THERMAL_EQUIPMENT_TAGS
        con_ontology = THERMAL_CONCEPT_TAGS
    else:
        eq_ontology = ELECTRICAL_EQUIPMENT_TAGS
        con_ontology = ELECTRICAL_CONCEPT_TAGS

    for tag, keywords in eq_ontology.items():
        if any(_keyword_in_text(kw, text_lower) for kw in keywords):
            equipment_tags.append(tag)

    for tag, keywords in con_ontology.items():
        if any(_keyword_in_text(kw, text_lower) for kw in keywords):
            concept_tags.append(tag)

    return TagResult(equipment_tags=equipment_tags, concept_tags=concept_tags)


def _keyword_in_text(keyword: str, text_lower: str) -> bool:
    """Check if a keyword phrase appears in the text (word-boundary aware)."""
    # Multi-word phrases: simple substring match
    if " " in keyword:
        return keyword in text_lower
    # Single words: use word boundary regex
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text_lower))


def get_all_thermal_tags() -> list[str]:
    """Return all canonical thermal tag names."""
    return list(ALL_THERMAL_TAGS.keys())


def get_all_electrical_tags() -> list[str]:
    """Return all canonical electrical tag names."""
    return list(ALL_ELECTRICAL_TAGS.keys())
