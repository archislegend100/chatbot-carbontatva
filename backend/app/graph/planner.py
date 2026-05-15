"""graph/planner.py — LLM-based query router with rule-based fallback."""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------
# PLANNER SYSTEM PROMPT (kept compact for low latency)
# -------------------------------------------------------

PLANNER_SYSTEM = """You are a routing planner for an industrial energy efficiency assistant.
Analyze the user query and output a JSON routing decision.
Be precise and fast. Use ONLY these exact values."""

PLANNER_SCHEMA = """{
  "query_type": "qa|explainer|troubleshoot|opportunity|comparison|navigation|checklist|summarize",
  "utility_domain": "thermal|electrical|both",
  "equipment_tags": ["list of equipment mentioned, max 5"],
  "candidate_tools": ["top 2 tool names"],
  "selected_tool": "best single tool name",
  "confidence": 0.95,
  "why_selected": "one sentence reason",
  "retrieval_profile": "semantic|table|list|metadata",
  "response_style": "direct|structured|bullet_list|comparison_table|checklist"
}"""

PLANNER_USER_TEMPLATE = """Query: "{query}"

Domain hint: {domain_hint}

Respond ONLY with valid JSON matching this schema:
{schema}

Tool guide:
- qa: factual questions ("what is", "how does")
- explainer: concept explanations ("explain", "describe in detail")  
- troubleshoot: problems/diagnosis ("why is", "not working", "low efficiency")
- opportunity: finding savings ("how to improve", "energy saving opportunities")
- comparison: comparing things ("compare", "vs", "difference between")
- navigation: finding sections ("which chapter", "where in the manual")
- checklist: audit/inspection lists ("checklist", "what to check", "audit")
- summarize: chapter/topic summaries ("summarize", "overview", "key points")"""


# -------------------------------------------------------
# RULE-BASED FALLBACK (fast, no LLM call)
# -------------------------------------------------------

_TOOL_KEYWORDS: dict[str, list[str]] = {
    "troubleshoot": ["problem", "issue", "fault", "failure", "not working", "low efficiency",
                     "why is", "high consumption", "excessive", "cause", "diagnose", "root cause"],
    "checklist": ["checklist", "audit checklist", "what to check", "inspection", "audit list"],
    "comparison": ["compare", "vs", "versus", "difference between", "which is better"],
    "opportunity": ["how to improve", "how to save", "energy saving", "opportunities",
                    "reduce consumption", "optimize", "best practice"],
    "navigation": ["which chapter", "which section", "where can i find", "find in book",
                   "page number", "table of contents"],
    "summarize": ["summarize", "summary", "overview of", "key points", "main points"],
    "explainer": ["explain", "elaborate", "clarify", "help me understand", "what does it mean",
                  "concept of", "theory of"],
    "qa": ["what is", "what are", "how does", "how do", "define", "when", "where"],
}

_THERMAL_KW = ["boiler", "furnace", "steam", "combustion", "fuel", "thermal", "heat",
                "flue gas", "fire", "chimney", "stack", "fbc", "dryer", "kiln",
                "condensate", "steam trap", "cogeneration", "waste heat", "insulation"]

_ELECTRICAL_KW = ["motor", "power factor", "capacitor", "transformer", "tariff",
                   "maximum demand", "kvar", "kwh", "kva", "electrical", "load",
                   "vfd", "drive", "compressed air", "hvac", "refrigeration", "chiller",
                   "pump", "fan", "lighting", "led", "dg set", "generator", "harmonic"]


def _rule_based_plan(query: str, domain_hint: Optional[str] = None) -> dict:
    """Fast rule-based fallback planner. No LLM needed."""
    q = query.lower()
    
    # Tool detection
    selected_tool = "qa"
    for tool, keywords in _TOOL_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            selected_tool = tool
            break
    
    # Domain detection
    if domain_hint and domain_hint in ("thermal", "electrical"):
        domain = domain_hint
    else:
        t_hits = sum(1 for kw in _THERMAL_KW if kw in q)
        e_hits = sum(1 for kw in _ELECTRICAL_KW if kw in q)
        if t_hits > e_hits:
            domain = "thermal"
        elif e_hits > t_hits:
            domain = "electrical"
        else:
            domain = "both"
    
    # Equipment tags
    eq_tags = [kw for kw in (_THERMAL_KW + _ELECTRICAL_KW) if kw in q][:5]
    
    # Retrieval profile
    profile_map = {
        "checklist": "list",
        "comparison": "semantic",
        "navigation": "metadata",
        "summarize": "semantic",
    }
    retrieval_profile = profile_map.get(selected_tool, "semantic")
    
    style_map = {
        "checklist": "checklist",
        "comparison": "comparison_table",
        "troubleshoot": "structured",
        "summarize": "bullet_list",
    }
    
    return {
        "query_type": selected_tool,
        "utility_domain": domain,
        "equipment_tags": eq_tags,
        "candidate_tools": [selected_tool, "qa"],
        "selected_tool": selected_tool,
        "confidence": 0.55,
        "why_selected": "Rule-based fallback classifier",
        "retrieval_profile": retrieval_profile,
        "response_style": style_map.get(selected_tool, "direct"),
    }


# -------------------------------------------------------
# LLM PLANNER
# -------------------------------------------------------

def build_planner_prompt(query: str, domain_hint: Optional[str] = None) -> tuple[str, str]:
    """Build the system + user prompts for the planner."""
    hint = domain_hint or "not specified — infer from query"
    user = PLANNER_USER_TEMPLATE.format(
        query=query,
        domain_hint=hint,
        schema=PLANNER_SCHEMA,
    )
    return PLANNER_SYSTEM, user


def parse_planner_response(raw: str) -> Optional[dict]:
    """
    Parse the planner's JSON response.
    
    With Ollama format=json, the response should be valid JSON directly.
    We still do a regex extraction as safety net.
    """
    if not raw:
        return None
    
    raw = raw.strip()
    
    # Direct JSON parse first (works with Ollama format=json)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "selected_tool" in data:
            return data
    except json.JSONDecodeError:
        pass
    
    # Regex fallback — extract JSON object from response
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "selected_tool" in data:
                return data
        except json.JSONDecodeError:
            pass
    
    return None


async def run_planner(
    query: str,
    llm_client,
    domain_hint: Optional[str] = None,
    timeout_s: float = 8.0,
) -> tuple[dict, str, bool]:
    """
    Run the structured planner.
    
    Returns:
        (planner_dict, raw_response, had_error)
    """
    t0 = time.monotonic()
    
    system_prompt, user_prompt = build_planner_prompt(query, domain_hint)
    had_error = False
    
    try:
        # Use JSON mode if available (Ollama-specific)
        raw = await llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
            temperature=0.0,
            json_mode=True,  # Ollama format=json
        )
        
        plan = parse_planner_response(raw)
        if plan is None:
            logger.warning("Planner JSON parse failed — using rule-based fallback")
            plan = _rule_based_plan(query, domain_hint)
            had_error = True
        
    except Exception as e:
        logger.warning(f"Planner LLM call failed ({e}) — using rule-based fallback")
        plan = _rule_based_plan(query, domain_hint)
        raw = json.dumps(plan)
        had_error = True
    
    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        f"Planner: tool={plan.get('selected_tool')} "
        f"domain={plan.get('utility_domain')} "
        f"conf={plan.get('confidence', 0):.2f} "
        f"({elapsed_ms:.0f}ms) {'[fallback]' if had_error else ''}"
    )
    
    return plan, raw, had_error
