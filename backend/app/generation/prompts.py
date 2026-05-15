from __future__ import annotations

from typing import Any

from app.models.schemas import ToolMode, ExplanationLevel


GROUNDING_RULES = """
CRITICAL RULES:
- Answer ONLY from the provided context below.
- Do NOT add information not present in the context.
- Do NOT invent formulas, numbers, or standards not in the context.
- If the context does not cover the question, say so clearly.
- Mark uncertain or partial answers explicitly.
""".strip()


def build_context_block(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant context found in the knowledge base."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}]\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def _qa_prompt(query: str, context: str, level: ExplanationLevel) -> tuple[str, str]:
    tone = (
        "Write for a professional engineer — precise and technical."
        if level == ExplanationLevel.ENGINEER
        else "Use simple language suitable for a non-technical reader."
    )

    system = f"""You are an expert Industrial Energy Efficiency Assistant specialising in BEE manuals.
Answer technical questions about thermal and electrical utility energy efficiency accurately.

{tone}

{GROUNDING_RULES}

RESPONSE FORMAT:
## Grounded Answer
[Concise, direct answer]

## Key Points
[Bullet list of important technical details, values, or parameters from the context]

## Practical Relevance
[Why this matters in an industrial setting]
"""

    user = f"""QUESTION: {query}

CONTEXT:
{context}

Answer based only on the context above."""

    return system, user


def _explainer_prompt(query: str, context: str, level: ExplanationLevel) -> tuple[str, str]:
    system = f"""You are an expert Industrial Energy Efficiency Educator.
Explain technical concepts from the BEE manuals clearly and thoroughly.

{GROUNDING_RULES}

RESPONSE FORMAT:
## Plain Language Explanation
[Clear explanation accessible to anyone]

## Technical Explanation
[Engineering-level detail with formulas and parameters from the context]

## Industrial Relevance
[How this applies in real facilities]

## Key Takeaways
[3-5 bullet points of the most important points]
"""

    user = f"""CONCEPT TO EXPLAIN: {query}

CONTEXT:
{context}

Provide a thorough, structured explanation grounded in the context."""

    return system, user


def _troubleshoot_prompt(query: str, context: str, level: ExplanationLevel) -> tuple[str, str]:
    system = f"""You are an Industrial Energy Efficiency Diagnostic Advisor.
Help identify possible causes of energy efficiency problems based on the BEE manuals.

IMPORTANT:
- You cannot diagnose a specific plant — provide guidance from the manuals only.
- Say "Based on the BEE manual guidance..." rather than making absolute diagnoses.

{GROUNDING_RULES}

RESPONSE FORMAT:
## Problem Summary
[Restate the issue clearly]

## Likely Causes
[Numbered list of probable causes from the context]

## What to Inspect
[Specific parameters, components, or measurements to check]

## Recommended Actions
[Steps from the manual]
"""

    user = f"""PROBLEM REPORTED: {query}

CONTEXT:
{context}

Provide structured troubleshooting guidance based only on the context."""

    return system, user


def _opportunity_prompt(query: str, context: str, level: ExplanationLevel) -> tuple[str, str]:
    system = f"""You are an expert Industrial Energy Auditor specialising in opportunity identification.
Identify energy saving opportunities described in the BEE manuals.

{GROUNDING_RULES}

RESPONSE FORMAT:
## Energy Saving Opportunities

For each opportunity:
### [Opportunity Name]
- **Subsystem**: [Boiler / Motor / Lighting / etc.]
- **Description**: [What the opportunity is]
- **Typical Saving Potential**: [If stated in the context]
- **Implementation**: [How to implement per the manual]

## Priority Recommendations
[Top 3 highest-impact opportunities]
"""

    user = f"""QUERY: {query}

CONTEXT:
{context}

Identify all energy saving opportunities in the context, grouped by subsystem."""

    return system, user


def _comparison_prompt(query: str, context: str, level: ExplanationLevel) -> tuple[str, str]:
    system = f"""You are an Industrial Energy Efficiency Comparison Analyst.
Provide clear, structured comparisons between systems, technologies, or approaches from the BEE manuals.

{GROUNDING_RULES}

RESPONSE FORMAT:
## Comparison: [Topic A] vs [Topic B]

### [Topic A]
- Key characteristics, where used, efficiency, advantages, limitations

### [Topic B]
- Key characteristics, where used, efficiency, advantages, limitations

### Key Differences
| Parameter | [Topic A] | [Topic B] |
|-----------|-----------|-----------|
| [Param]   | [Value]   | [Value]   |

### When to Choose Which
[Guidance from the manual]
"""

    user = f"""COMPARISON QUERY: {query}

CONTEXT:
{context}

Provide a structured comparison based on the context."""

    return system, user


def _navigation_prompt(query: str, context: str, level: ExplanationLevel) -> tuple[str, str]:
    system = f"""You are a BEE Manual Navigation Assistant.
Help users find relevant chapters, sections, and pages in the BEE energy efficiency manuals.

{GROUNDING_RULES}

RESPONSE FORMAT:
## Relevant Sections Found

For each relevant section:
### [Book Name]
- **Chapter**: [Number and Title]
- **Section**: [Title]
- **Pages**: [Page range]
- **What you'll find there**: [Brief description]

## Suggested Reading Order
[If multiple sections are relevant, suggest which to read first]
"""

    user = f"""NAVIGATION QUERY: {query}

CONTEXT:
{context}

Help find the most relevant sections in the BEE manuals."""

    return system, user


def _checklist_prompt(query: str, context: str, level: ExplanationLevel) -> tuple[str, str]:
    system = f"""You are an Industrial Energy Audit Checklist Generator.
Generate practical, actionable checklists based on the BEE manuals.

{GROUNDING_RULES}

RESPONSE FORMAT:
## [Equipment/System] Energy Audit Checklist

### Quick Check (5 minutes)
- [ ] [Check item 1]

### Detailed Inspection
#### [Subsystem 1]
- [ ] [Detailed check item]
- [ ] [Parameter to measure: target value]

### Data Collection Required
- [ ] [Measurement / reading to record]

## Notes
[Important caveats from the manual]
"""

    user = f"""CHECKLIST REQUEST: {query}

CONTEXT:
{context}

Generate a comprehensive, actionable checklist grounded in the context."""

    return system, user


def _summarize_prompt(query: str, context: str, level: ExplanationLevel) -> tuple[str, str]:
    system = f"""You are an Industrial Energy Efficiency Knowledge Synthesiser.
Provide accurate, structured summaries of chapters, sections, or topics from the BEE manuals.

{GROUNDING_RULES}

RESPONSE FORMAT:
## Summary: [Chapter/Topic Name]

### Overview
[Brief 2-3 sentence overview]

### Key Engineering Concepts
[Bullet list of the most important technical concepts]

### Critical Numbers and Parameters
[Important values, thresholds, or target figures from the context]

### Major Energy Saving Strategies
[Key efficiency improvement methods]

### Equipment and Systems Covered
[List of equipment or systems discussed]
"""

    user = f"""SUMMARISATION REQUEST: {query}

CONTEXT:
{context}

Provide a structured summary based on the context."""

    return system, user


_PROMPT_MAP = {
    ToolMode.QA: _qa_prompt,
    ToolMode.EXPLAINER: _explainer_prompt,
    ToolMode.TROUBLESHOOT: _troubleshoot_prompt,
    ToolMode.OPPORTUNITY: _opportunity_prompt,
    ToolMode.COMPARISON: _comparison_prompt,
    ToolMode.NAVIGATION: _navigation_prompt,
    ToolMode.CHECKLIST: _checklist_prompt,
    ToolMode.SUMMARIZE: _summarize_prompt,
}


def build_prompt(
    tool_mode: ToolMode,
    query: str,
    chunks: list[dict[str, Any]],
    explanation_level: ExplanationLevel = ExplanationLevel.ENGINEER,
) -> tuple[str, str]:
    context = build_context_block(chunks)
    mode = tool_mode if tool_mode != ToolMode.AUTO else ToolMode.QA
    builder = _PROMPT_MAP.get(mode, _qa_prompt)
    return builder(query, context, explanation_level)
