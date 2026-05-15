#!/usr/bin/env python3
"""
scripts/eval_queries.py
=======================
Evaluation query set for the Industrial Energy Efficiency Copilot.

Runs a set of representative queries across all tool modes and domains
and prints the results for quality evaluation.

Usage:
    python scripts/eval_queries.py
    python scripts/eval_queries.py --tool qa
    python scripts/eval_queries.py --domain thermal
    python scripts/eval_queries.py --max-queries 5

This script requires:
- Backend indexes to be built (run scripts/ingest.py first)
- LLM API key configured in .env
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Evaluation query set
EVAL_QUERIES = [
    # ---- QA TOOL ----
    {
        "tool": "qa", "domain": "thermal",
        "query": "What is the optimum excess air percentage for a coal-fired boiler?",
        "expected_keywords": ["excess air", "coal", "combustion", "efficiency"],
    },
    {
        "tool": "qa", "domain": "thermal",
        "query": "What is the enthalpy of steam at 10 bar pressure?",
        "expected_keywords": ["enthalpy", "steam", "pressure", "bar"],
    },
    {
        "tool": "qa", "domain": "electrical",
        "query": "What is the definition of load factor in electrical systems?",
        "expected_keywords": ["load factor", "maximum demand", "average load"],
    },
    {
        "tool": "qa", "domain": "electrical",
        "query": "What is the penalty for low power factor in HT consumers?",
        "expected_keywords": ["power factor", "penalty", "tariff", "reactive"],
    },
    # ---- EXPLAINER TOOL ----
    {
        "tool": "explainer", "domain": "thermal",
        "query": "Explain the concept of pinch analysis in heat exchanger networks",
        "expected_keywords": ["pinch", "heat exchanger", "minimum utility", "temperature"],
    },
    {
        "tool": "explainer", "domain": "electrical",
        "query": "Explain how a VFD (Variable Frequency Drive) saves energy in centrifugal pumps",
        "expected_keywords": ["VFD", "frequency", "speed", "affinity law", "pump"],
    },
    # ---- TROUBLESHOOT TOOL ----
    {
        "tool": "troubleshoot", "domain": "thermal",
        "query": "Why is my boiler efficiency dropping below 75%?",
        "expected_keywords": ["efficiency", "excess air", "flue gas", "insulation", "blowdown"],
    },
    {
        "tool": "troubleshoot", "domain": "electrical",
        "query": "Our compressed air system has high energy consumption - what should I check?",
        "expected_keywords": ["compressed air", "leakage", "pressure", "compressor"],
    },
    # ---- OPPORTUNITY TOOL ----
    {
        "tool": "opportunity", "domain": "thermal",
        "query": "What are the main energy saving opportunities in a steam distribution system?",
        "expected_keywords": ["steam trap", "insulation", "condensate", "flash steam", "saving"],
    },
    {
        "tool": "opportunity", "domain": "electrical",
        "query": "What are energy saving opportunities for industrial motors?",
        "expected_keywords": ["motor", "efficiency", "VFD", "rewinding", "IE3"],
    },
    # ---- COMPARISON TOOL ----
    {
        "tool": "comparison", "domain": "thermal",
        "query": "Compare fire tube boilers and water tube boilers",
        "expected_keywords": ["fire tube", "water tube", "capacity", "pressure", "efficiency"],
    },
    {
        "tool": "comparison", "domain": "electrical",
        "query": "Compare DOL starter vs star-delta starter vs VFD for motor starting",
        "expected_keywords": ["DOL", "star-delta", "VFD", "starting current", "motor"],
    },
    # ---- NAVIGATION TOOL ----
    {
        "tool": "navigation", "domain": "thermal",
        "query": "Which chapter covers waste heat recovery in the thermal manual?",
        "expected_keywords": ["chapter", "waste heat", "recovery", "section"],
    },
    {
        "tool": "navigation", "domain": "electrical",
        "query": "Where can I find information about power factor correction?",
        "expected_keywords": ["chapter", "section", "power factor", "capacitor"],
    },
    # ---- CHECKLIST TOOL ----
    {
        "tool": "checklist", "domain": "thermal",
        "query": "Generate an energy audit checklist for industrial boilers",
        "expected_keywords": ["boiler", "checklist", "excess air", "flue gas", "insulation"],
    },
    {
        "tool": "checklist", "domain": "electrical",
        "query": "Create an inspection checklist for cooling towers",
        "expected_keywords": ["cooling tower", "check", "temperature", "flow", "drift"],
    },
    # ---- SUMMARIZE TOOL ----
    {
        "tool": "summarize", "domain": "thermal",
        "query": "Summarize the key points about cogeneration systems",
        "expected_keywords": ["cogeneration", "thermal", "electrical", "topping", "bottoming"],
    },
    {
        "tool": "summarize", "domain": "electrical",
        "query": "Summarize energy conservation measures for lighting systems",
        "expected_keywords": ["lighting", "LED", "luminaire", "lux", "energy saving"],
    },
    # ---- CROSS-DOMAIN ----
    {
        "tool": "qa", "domain": None,
        "query": "What is the difference between energy audit and energy management?",
        "expected_keywords": ["audit", "management", "energy", "assessment"],
    },
    {
        "tool": "opportunity", "domain": None,
        "query": "What are the top energy conservation opportunities in a manufacturing plant?",
        "expected_keywords": ["opportunity", "saving", "efficiency"],
    },
]


async def run_query(query_def: dict) -> dict:
    """Run a single eval query against the live backend."""
    import httpx

    url = "http://localhost:8000/api/query"
    payload = {
        "query": query_def["query"],
        "tool_mode": query_def["tool"],
        "domain_filter": query_def.get("domain"),
        "explanation_level": "engineer",
    }

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        latency = time.time() - t0

        # Simple keyword check
        answer_lower = data.get("answer", "").lower()
        keywords_found = [
            kw for kw in query_def.get("expected_keywords", [])
            if kw.lower() in answer_lower
        ]
        kw_score = len(keywords_found) / max(len(query_def.get("expected_keywords", [1])), 1)

        return {
            "query": query_def["query"],
            "tool": query_def["tool"],
            "domain": query_def.get("domain"),
            "status": "ok",
            "latency_s": round(latency, 2),
            "citations": len(data.get("citations", [])),
            "kw_score": round(kw_score, 2),
            "keywords_found": keywords_found,
            "answer_preview": data.get("answer", "")[:200],
        }
    except Exception as e:
        return {
            "query": query_def["query"],
            "tool": query_def["tool"],
            "domain": query_def.get("domain"),
            "status": "error",
            "error": str(e),
        }


async def main():
    parser = argparse.ArgumentParser(description="Run evaluation queries")
    parser.add_argument("--tool", help="Filter by tool mode")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--max-queries", type=int, help="Max queries to run")
    args = parser.parse_args()

    queries = EVAL_QUERIES
    if args.tool:
        queries = [q for q in queries if q["tool"] == args.tool]
    if args.domain:
        queries = [q for q in queries if q.get("domain") == args.domain]
    if args.max_queries:
        queries = queries[:args.max_queries]

    print(f"\nRunning {len(queries)} evaluation queries against http://localhost:8000...\n")

    results = []
    for i, qdef in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] [{qdef['tool']}] [{qdef.get('domain', 'all')}]")
        print(f"  Q: {qdef['query'][:80]}...")
        result = await run_query(qdef)
        results.append(result)

        if result["status"] == "ok":
            print(f"  ✅ {result['latency_s']}s | {result['citations']} citations | kw_score={result['kw_score']}")
        else:
            print(f"  ❌ Error: {result.get('error')}")

        print()

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"  Total: {len(results)} | OK: {len(ok)} | Errors: {len(results) - len(ok)}")
    if ok:
        avg_latency = sum(r["latency_s"] for r in ok) / len(ok)
        avg_kw = sum(r["kw_score"] for r in ok) / len(ok)
        avg_citations = sum(r["citations"] for r in ok) / len(ok)
        print(f"  Avg latency: {avg_latency:.1f}s")
        print(f"  Avg keyword score: {avg_kw:.2f}")
        print(f"  Avg citations: {avg_citations:.1f}")

    # Save results
    output = PROJECT_ROOT / "artifacts" / "eval_results.json"
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    asyncio.run(main())
