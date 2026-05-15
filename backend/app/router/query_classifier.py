"""
router/query_classifier.py
===========================
Query classifier for the Industrial Energy Efficiency Copilot.

This module classifies incoming queries to determine:
1. TOOL MODE — which of the 8 tools should handle this query
2. UTILITY DOMAIN — thermal, electrical, or both
3. EQUIPMENT TAGS — relevant equipment mentioned/implied
4. CONCEPT TAGS — relevant concepts mentioned/implied

Classification uses a two-pass strategy:
  Pass 1: Rule-based keyword matching (fast, no LLM needed)
  Pass 2: LLM-based classification for ambiguous queries

This design keeps latency low for obvious queries and uses the LLM
only when needed.

Usage:
    classifier = QueryClassifier(llm_client)
    result = classifier.classify("How do I improve boiler efficiency?")
    # result.tool_mode == ToolMode.QA
    # result.utility_domain == UtilityDomain.THERMAL
"""

from __future__ import annotations

import json
import re
from typing import Optional

from app.models.schemas import (
    ClassificationResult,
    ToolMode,
    UtilityDomain,
)
from app.ingestion.ontology import (
    ALL_THERMAL_TAGS,
    ALL_ELECTRICAL_TAGS,
    detect_tags,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# RULE-BASED PATTERNS
# ============================================================

# Keywords that strongly suggest each tool mode
TOOL_KEYWORDS: dict[ToolMode, list[str]] = {
    ToolMode.QA: [
        "what is", "what are", "how does", "how do", "why", "when", "where",
        "define", "explain what", "tell me about", "describe",
    ],
    ToolMode.EXPLAINER: [
        "explain", "elaborate", "clarify", "help me understand", "in simple terms",
        "what does it mean", "break down", "concept of", "theory of",
    ],
    ToolMode.TROUBLESHOOT: [
        "problem", "issue", "fault", "failure", "not working", "low efficiency",
        "high consumption", "excessive", "cause", "why is", "troubleshoot",
        "diagnose", "root cause", "breakdown", "malfunction",
    ],
    ToolMode.OPPORTUNITY: [
        "how to improve", "how to save", "energy saving", "opportunities",
        "reduce consumption", "optimize", "improve efficiency", "potential saving",
        "recommendations", "best practice", "conservation measures",
    ],
    ToolMode.COMPARISON: [
        "compare", "comparison", "difference between", "vs", "versus",
        "which is better", "pros and cons", "advantages of", "disadvantages of",
        "contrast",
    ],
    ToolMode.NAVIGATION: [
        "which chapter", "which section", "where can I find", "find in book",
        "page number", "table of contents", "where is", "what chapter",
        "navigate", "index",
    ],
    ToolMode.CHECKLIST: [
        "checklist", "check list", "inspection list", "audit checklist",
        "list of checks", "what to check", "inspection points", "audit steps",
    ],
    ToolMode.SUMMARIZE: [
        "summarize", "summary", "overview of", "brief description",
        "key points", "main points", "chapter summary", "what are the main",
        "highlight",
    ],
}

# Keywords that suggest thermal vs electrical domain
THERMAL_DOMAIN_KEYWORDS = [
    "boiler", "furnace", "steam", "combustion", "fuel", "thermal",
    "heat", "insulation", "refractory", "cogeneration", "waste heat",
    "flue gas", "fire", "chimney", "stack", "fbc", "dryer", "kiln",
    "recuperator", "economiser", "condensate", "steam trap",
]

ELECTRICAL_DOMAIN_KEYWORDS = [
    "motor", "power factor", "capacitor", "transformer", "tariff",
    "maximum demand", "kvar", "kwh", "kva", "electrical", "load",
    "vfd", "drive", "compressed air", "hvac", "refrigeration", "chiller",
    "pump", "fan", "lighting", "led", "dg set", "generator", "harmonic",
    "ecbc",
]


# ============================================================
# CLASSIFIER
# ============================================================


class QueryClassifier:
    """
    Classifies queries into tool mode, domain, and tags.

    Args:
        llm_client: Optional LLM client for fallback LLM classification.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a query.

        Args:
            query: The user's raw query string.

        Returns:
            ClassificationResult with tool_mode, domain, tags, confidence.
        """
        query_lower = query.lower().strip()

        # Pass 1: Rule-based
        tool_mode, tool_confidence = self._rule_based_tool(query_lower)
        domain, domain_confidence = self._rule_based_domain(query_lower)

        # Detect equipment and concept tags from query
        eq_tags: list[str] = []
        con_tags: list[str] = []
        if domain in (UtilityDomain.THERMAL, UtilityDomain.UNKNOWN):
            r = detect_tags(query, "thermal")
            eq_tags.extend(r.equipment_tags)
            con_tags.extend(r.concept_tags)
        if domain in (UtilityDomain.ELECTRICAL, UtilityDomain.UNKNOWN):
            r = detect_tags(query, "electrical")
            eq_tags.extend(r.equipment_tags)
            con_tags.extend(r.concept_tags)

        overall_confidence = (tool_confidence + domain_confidence) / 2.0

        # Pass 2: LLM fallback for ambiguous queries
        if overall_confidence < 0.4 and self.llm_client is not None:
            try:
                llm_result = self._llm_classify(query)
                if llm_result:
                    llm_result.equipment_tags = list(set(eq_tags + llm_result.equipment_tags))
                    llm_result.concept_tags = list(set(con_tags + llm_result.concept_tags))
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}. Using rule-based result.")

        return ClassificationResult(
            tool_mode=tool_mode,
            utility_domain=domain if domain != UtilityDomain.UNKNOWN else None,
            equipment_tags=list(set(eq_tags)),
            concept_tags=list(set(con_tags)),
            confidence=overall_confidence,
            reasoning="rule-based",
        )

    def _rule_based_tool(self, query_lower: str) -> tuple[ToolMode, float]:
        """Score each tool mode by keyword matches."""
        scores: dict[ToolMode, int] = {mode: 0 for mode in ToolMode}

        for mode, keywords in TOOL_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[mode] += 1

        best_mode = max(scores, key=lambda m: scores[m])
        best_score = scores[best_mode]

        if best_score == 0:
            return ToolMode.QA, 0.3  # Default to QA

        confidence = min(1.0, best_score * 0.2 + 0.4)
        return best_mode, confidence

    def _rule_based_domain(self, query_lower: str) -> tuple[UtilityDomain, float]:
        """Detect likely domain from query keywords."""
        thermal_hits = sum(1 for kw in THERMAL_DOMAIN_KEYWORDS if kw in query_lower)
        electrical_hits = sum(1 for kw in ELECTRICAL_DOMAIN_KEYWORDS if kw in query_lower)

        total = thermal_hits + electrical_hits
        if total == 0:
            return UtilityDomain.UNKNOWN, 0.3

        if thermal_hits > electrical_hits:
            confidence = min(1.0, thermal_hits / max(total, 1) * 0.8 + 0.3)
            return UtilityDomain.THERMAL, confidence
        elif electrical_hits > thermal_hits:
            confidence = min(1.0, electrical_hits / max(total, 1) * 0.8 + 0.3)
            return UtilityDomain.ELECTRICAL, confidence
        else:
            return UtilityDomain.UNKNOWN, 0.4

    def _llm_classify(self, query: str) -> Optional[ClassificationResult]:
        """Use LLM to classify ambiguous query."""
        prompt = f"""You are a classifier for an industrial energy efficiency assistant.

Classify this user query:
"{query}"

Respond ONLY with a JSON object with these exact fields:
{{
  "tool_mode": "qa" | "explainer" | "troubleshoot" | "opportunity" | "comparison" | "navigation" | "checklist" | "summarize",
  "utility_domain": "thermal" | "electrical" | null,
  "equipment_tags": ["list", "of", "relevant", "equipment"],
  "concept_tags": ["list", "of", "relevant", "concepts"],
  "reasoning": "brief explanation"
}}

Tool modes:
- qa: direct factual questions
- explainer: explaining concepts in depth
- troubleshoot: diagnosing problems or failures
- opportunity: finding energy saving opportunities
- comparison: comparing two systems or approaches
- navigation: finding chapters/sections
- checklist: generating inspection/audit checklists
- summarize: summarizing chapters or topics
"""
        try:
            response_text = self.llm_client.generate_sync(
                prompt=prompt,
                max_tokens=300,
                temperature=0.0,
            )
            # Parse JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return ClassificationResult(
                    tool_mode=ToolMode(data.get("tool_mode", "qa")),
                    utility_domain=(
                        UtilityDomain(data["utility_domain"])
                        if data.get("utility_domain")
                        else None
                    ),
                    equipment_tags=data.get("equipment_tags", []),
                    concept_tags=data.get("concept_tags", []),
                    confidence=0.8,
                    reasoning=data.get("reasoning", "llm-based"),
                )
        except Exception as e:
            logger.warning(f"LLM classify parse error: {e}")
        return None
