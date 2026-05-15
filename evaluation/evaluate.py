#!/usr/bin/env python3
"""
evaluation/evaluate.py
=======================
Comprehensive evaluation script for the Industrial Energy Efficiency Copilot.

Measures:
  - Retrieval quality (Recall@K, Precision@K, MRR, keyword coverage)
  - Router/planner accuracy (tool selection, domain accuracy)
  - Answer quality (semantic similarity, completeness, keyword overlap)
  - Citation quality (presence, completeness, source correctness)
  - Latency (per-stage and end-to-end)
  - End-to-end usefulness score

Output:
  - JSONL results file
  - CSV summary
  - Markdown report
  - Console summary

Usage:
  python evaluation/evaluate.py --base-url http://localhost:8000 --k 5
  python evaluation/evaluate.py --dataset evaluation/synthetic_dataset.json
  python evaluation/evaluate.py --category troubleshoot --outdir evaluation/results/run_01
  python evaluation/evaluate.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

import httpx

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TestCase:
    id: str
    category: str
    query: str
    expected_tool: str
    expected_domain: str
    expected_keywords: list[str]
    expected_source_hints: list[str]
    reference_answer: str
    expected_structure: str


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    keyword_coverage: float = 0.0
    retrieved_count: int = 0
    k: int = 5


@dataclass
class RouterMetrics:
    tool_correct: bool = False
    domain_correct: bool = False
    planner_confidence: float = 0.0
    planner_fallback: bool = False
    predicted_tool: str = ""
    predicted_domain: str = ""


@dataclass
class AnswerMetrics:
    keyword_overlap: float = 0.0
    completeness_score: float = 0.0
    structure_adherence: float = 0.0
    answer_length: int = 0
    answer_non_empty: bool = False


@dataclass
class CitationMetrics:
    citations_present: bool = False
    citation_count: int = 0
    avg_field_completeness: float = 0.0
    source_hint_match: float = 0.0


@dataclass
class LatencyMetrics:
    planner_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    api_call_ms: float = 0.0


@dataclass
class TestResult:
    test_id: str
    category: str
    query: str
    expected_tool: str
    expected_domain: str
    predicted_tool: str
    predicted_domain: str
    answer: str
    planner_raw: str
    retrieval: RetrievalMetrics
    router: RouterMetrics
    answer_metrics: AnswerMetrics
    citation: CitationMetrics
    latency: LatencyMetrics
    composite_score: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_retrieval_metrics(
    chunks: list[dict],
    expected_keywords: list[str],
    expected_source_hints: list[str],
    k: int,
) -> RetrievalMetrics:
    """
    Compute retrieval metrics.

    Recall@K: fraction of expected source hints found in top-K chunks.
    Precision@K: fraction of top-K chunks that contain any expected keyword.
    MRR: 1 / rank of first relevant chunk (keyword or source match).
    Keyword coverage: fraction of expected keywords found anywhere in chunks.
    """
    if not chunks:
        return RetrievalMetrics(k=k, retrieved_count=0)

    top_k = chunks[:k]

    # ---- Recall@K: source hint coverage ----
    if expected_source_hints:
        matched_hints = set()
        for chunk in top_k:
            chunk_text = _chunk_text(chunk).lower()
            for hint in expected_source_hints:
                if hint.lower() in chunk_text:
                    matched_hints.add(hint.lower())
        recall = len(matched_hints) / len(expected_source_hints)
    else:
        recall = 1.0

    # ---- Precision@K: keyword relevance ----
    relevant_in_top_k = 0
    for chunk in top_k:
        chunk_text = _chunk_text(chunk).lower()
        if any(kw.lower() in chunk_text for kw in expected_keywords):
            relevant_in_top_k += 1
    precision = relevant_in_top_k / len(top_k) if top_k else 0.0

    # ---- MRR: rank of first relevant chunk ----
    mrr = 0.0
    for rank, chunk in enumerate(chunks, start=1):
        chunk_text = _chunk_text(chunk).lower()
        is_relevant = (
            any(kw.lower() in chunk_text for kw in expected_keywords) or
            any(hint.lower() in chunk_text for hint in expected_source_hints)
        )
        if is_relevant:
            mrr = 1.0 / rank
            break

    # ---- Keyword coverage across ALL retrieved chunks ----
    all_text = " ".join(_chunk_text(c).lower() for c in chunks)
    covered = sum(1 for kw in expected_keywords if kw.lower() in all_text)
    kw_coverage = covered / len(expected_keywords) if expected_keywords else 0.0

    return RetrievalMetrics(
        recall_at_k=round(recall, 4),
        precision_at_k=round(precision, 4),
        mrr=round(mrr, 4),
        keyword_coverage=round(kw_coverage, 4),
        retrieved_count=len(chunks),
        k=k,
    )


def compute_router_metrics(
    predicted_tool: str,
    predicted_domain: str,
    expected_tool: str,
    expected_domain: str,
    planner: dict,
) -> RouterMetrics:
    """Tool and domain routing accuracy."""
    tool_correct = predicted_tool.lower() == expected_tool.lower()

    # Domain matching — 'both' is acceptable if any expected domain is covered
    pred_domain = predicted_domain.lower() if predicted_domain else ""
    exp_domain = expected_domain.lower()
    domain_correct = (pred_domain == exp_domain or pred_domain == "both")

    confidence = float(planner.get("confidence", 0.0)) if planner else 0.0
    fallback = bool(planner.get("planner_error", False)) if planner else False

    return RouterMetrics(
        tool_correct=tool_correct,
        domain_correct=domain_correct,
        planner_confidence=round(confidence, 4),
        planner_fallback=fallback,
        predicted_tool=predicted_tool,
        predicted_domain=predicted_domain,
    )


def compute_answer_metrics(
    answer: str,
    reference: str,
    expected_keywords: list[str],
    expected_structure: str,
) -> AnswerMetrics:
    """
    Lightweight answer quality metrics.

    Uses token overlap (no external embeddings required).
    Completeness: ratio of reference tokens found in answer.
    Keyword overlap: fraction of expected keywords in answer.
    Structure adherence: checks if expected formatting signals are present.
    """
    if not answer:
        return AnswerMetrics(answer_non_empty=False)

    answer_lower = answer.lower()
    answer_tokens = set(_tokenise(answer_lower))
    ref_tokens = set(_tokenise(reference.lower()))

    # ---- Completeness: reference token coverage ----
    if ref_tokens:
        covered_ref = sum(1 for t in ref_tokens if t in answer_tokens)
        completeness = covered_ref / len(ref_tokens)
    else:
        completeness = 0.0

    # ---- Keyword overlap ----
    covered_kw = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    kw_overlap = covered_kw / len(expected_keywords) if expected_keywords else 0.0

    # ---- Structure adherence (heuristic) ----
    structure_score = _check_structure(answer, expected_structure)

    return AnswerMetrics(
        keyword_overlap=round(kw_overlap, 4),
        completeness_score=round(completeness, 4),
        structure_adherence=round(structure_score, 4),
        answer_length=len(answer.split()),
        answer_non_empty=len(answer.strip()) > 20,
    )


def compute_citation_metrics(
    citations: list[dict],
    expected_source_hints: list[str],
) -> CitationMetrics:
    """Citation presence, completeness, and source correctness."""
    if not citations:
        return CitationMetrics(citations_present=False)

    # ---- Field completeness ----
    required_fields = ["chunk_id", "book_name", "page_start", "relevance_score"]
    completeness_scores = []
    for c in citations:
        filled = sum(1 for f in required_fields if c.get(f) is not None)
        completeness_scores.append(filled / len(required_fields))
    avg_completeness = sum(completeness_scores) / len(completeness_scores)

    # ---- Source hint match ----
    all_citation_text = " ".join(
        f"{c.get('book_name', '')} {c.get('chapter_title', '')} {c.get('section_title', '')} {c.get('snippet', '')}"
        for c in citations
    ).lower()
    if expected_source_hints:
        matched = sum(1 for h in expected_source_hints if h.lower() in all_citation_text)
        hint_match = matched / len(expected_source_hints)
    else:
        hint_match = 1.0

    return CitationMetrics(
        citations_present=True,
        citation_count=len(citations),
        avg_field_completeness=round(avg_completeness, 4),
        source_hint_match=round(hint_match, 4),
    )


def compute_composite_score(
    retrieval: RetrievalMetrics,
    router: RouterMetrics,
    answer: AnswerMetrics,
    citation: CitationMetrics,
) -> float:
    """Weighted composite score (0–1)."""
    # Weights: sum = 1.0
    w = {"retrieval": 0.30, "router": 0.25, "answer": 0.30, "citation": 0.15}

    retrieval_score = (
        retrieval.recall_at_k * 0.4 +
        retrieval.precision_at_k * 0.3 +
        retrieval.mrr * 0.2 +
        retrieval.keyword_coverage * 0.1
    )
    router_score = (
        (1.0 if router.tool_correct else 0.0) * 0.6 +
        (1.0 if router.domain_correct else 0.0) * 0.4
    )
    answer_score = (
        answer.keyword_overlap * 0.4 +
        answer.completeness_score * 0.4 +
        answer.structure_adherence * 0.2
    )
    citation_score = (
        (1.0 if citation.citations_present else 0.0) * 0.4 +
        citation.avg_field_completeness * 0.3 +
        citation.source_hint_match * 0.3
    )

    composite = (
        retrieval_score * w["retrieval"] +
        router_score * w["router"] +
        answer_score * w["answer"] +
        citation_score * w["citation"]
    )
    return round(composite, 4)


# ============================================================
# HELPERS
# ============================================================

def _chunk_text(chunk: dict) -> str:
    """Extract text from a chunk dict (handles multiple schema variants)."""
    return (
        chunk.get("text") or
        chunk.get("snippet") or
        chunk.get("content") or
        str(chunk)
    )


def _tokenise(text: str) -> list[str]:
    """Simple whitespace+punctuation tokeniser."""
    return re.findall(r'\b[a-z][a-z0-9\-]*\b', text.lower())


def _check_structure(answer: str, expected_structure: str) -> float:
    """
    Heuristic structure adherence check.
    Returns 0–1 score based on pattern matching.
    """
    a = answer.lower()
    checks = {
        "direct": lambda: len(answer.split()) > 20,  # At least a paragraph
        "structured": lambda: any(m in a for m in ["1.", "2.", "3.", "##", "- ", "•"]),
        "structured_diagnosis": lambda: any(m in a for m in ["cause", "solution", "fix", "check", "1.", "2."]),
        "bullet_list": lambda: answer.count("\n") >= 2 or any(m in a for m in ["- ", "•", "1.", "2."]),
        "comparison_table": lambda: any(m in a for m in ["vs", "compare", "|", "whereas", "however"]),
        "checklist": lambda: any(m in a for m in ["1.", "2.", "3.", "☐", "[ ]", "check"]),
    }
    checker = checks.get(expected_structure)
    if checker:
        try:
            return 1.0 if checker() else 0.5
        except Exception:
            return 0.5
    return 0.75  # Unknown structure — give partial credit


# ============================================================
# API CLIENT
# ============================================================

def query_backend(
    base_url: str,
    test_case: TestCase,
    timeout: float = 120.0,
) -> dict:
    """
    Query the /api/query endpoint and capture full response.

    Returns raw response dict with timings added.
    """
    url = f"{base_url.rstrip('/')}/api/query"
    payload = {
        "query": test_case.query,
        "tool_mode": "auto",
        "domain_filter": None,
        "explanation_level": "engineer",
    }

    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            api_ms = (time.monotonic() - t0) * 1000
            data = resp.json()
            data["_api_call_ms"] = round(api_ms, 1)
            return data
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to backend at {base_url}. "
            "Start it with: ./run_backend.sh"
        )
    except httpx.TimeoutException:
        raise RuntimeError(f"Request timed out after {timeout}s")
    except Exception as e:
        raise RuntimeError(f"API error: {e}")


def extract_fields(response: dict, k: int) -> tuple:
    """
    Extract standardised fields from API response.
    Handles both v1 (tool_router) and v2 (LangGraph) response schemas.
    """
    answer = response.get("answer", "")
    citations = response.get("citations", [])
    classification = response.get("classification", {})
    node_latency = response.get("node_latency", {})

    predicted_tool = (
        classification.get("tool_mode") or
        response.get("tool_mode") or
        ""
    )
    predicted_domain = (
        classification.get("utility_domain") or
        response.get("utility_domain") or
        "both"
    )
    planner_confidence = classification.get("confidence", 0.0)
    planner_raw = classification.get("planner_raw", "{}")
    planner_error = classification.get("planner_error", False)

    planner_dict = {
        "confidence": planner_confidence,
        "planner_error": planner_error,
    }

    # Chunks from citations (or direct if available)
    chunks = citations  # Citations ARE the retrieved+reranked chunks in our schema

    # Latency extraction — try node_latency first, then top-level
    lat_ms = response.get("latency_ms", response.get("_api_call_ms", 0))
    planner_ms = node_latency.get("planner_router", 0)
    retrieval_ms = (
        node_latency.get("retrieve_dense", 0) +
        node_latency.get("retrieve_sparse", 0) +
        node_latency.get("merge_results", 0)
    )
    rerank_ms = node_latency.get("rerank", 0)
    generation_ms = node_latency.get("answer_generation", 0)

    latency = LatencyMetrics(
        planner_ms=round(planner_ms, 1),
        retrieval_ms=round(retrieval_ms, 1),
        rerank_ms=round(rerank_ms, 1),
        generation_ms=round(generation_ms, 1),
        total_ms=round(lat_ms, 1),
        api_call_ms=round(response.get("_api_call_ms", lat_ms), 1),
    )

    return answer, citations, chunks, predicted_tool, predicted_domain, planner_dict, planner_raw, latency


# ============================================================
# SINGLE CASE EVALUATION
# ============================================================

def evaluate_one(
    test_case: TestCase,
    base_url: str,
    k: int,
    verbose: bool = False,
) -> TestResult:
    """Run a single test case and compute all metrics."""
    ts = datetime.now().isoformat()

    try:
        response = query_backend(base_url, test_case)
        (
            answer, citations, chunks, predicted_tool, predicted_domain,
            planner_dict, planner_raw, latency
        ) = extract_fields(response, k)

        retrieval = compute_retrieval_metrics(chunks, test_case.expected_keywords, test_case.expected_source_hints, k)
        router = compute_router_metrics(predicted_tool, predicted_domain, test_case.expected_tool, test_case.expected_domain, planner_dict)
        answer_m = compute_answer_metrics(answer, test_case.reference_answer, test_case.expected_keywords, test_case.expected_structure)
        citation_m = compute_citation_metrics(citations, test_case.expected_source_hints)
        composite = compute_composite_score(retrieval, router, answer_m, citation_m)

        result = TestResult(
            test_id=test_case.id,
            category=test_case.category,
            query=test_case.query,
            expected_tool=test_case.expected_tool,
            expected_domain=test_case.expected_domain,
            predicted_tool=predicted_tool,
            predicted_domain=predicted_domain,
            answer=answer,
            planner_raw=planner_raw,
            retrieval=retrieval,
            router=router,
            answer_metrics=answer_m,
            citation=citation_m,
            latency=latency,
            composite_score=composite,
            timestamp=ts,
        )

    except Exception as e:
        result = TestResult(
            test_id=test_case.id,
            category=test_case.category,
            query=test_case.query,
            expected_tool=test_case.expected_tool,
            expected_domain=test_case.expected_domain,
            predicted_tool="ERROR",
            predicted_domain="ERROR",
            answer="",
            planner_raw="",
            retrieval=RetrievalMetrics(),
            router=RouterMetrics(),
            answer_metrics=AnswerMetrics(),
            citation=CitationMetrics(),
            latency=LatencyMetrics(),
            composite_score=0.0,
            error=str(e),
            timestamp=ts,
        )

    if verbose:
        status = "✅" if result.composite_score >= 0.5 else "⚠️" if result.composite_score >= 0.3 else "❌"
        tool_ok = "✓" if result.router.tool_correct else "✗"
        print(
            f"  {status} [{test_case.category:12s}] {test_case.id:30s} "
            f"score={result.composite_score:.2f}  "
            f"tool={tool_ok}({result.predicted_tool:15s})  "
            f"latency={result.latency.total_ms:.0f}ms"
        )
        if result.error:
            print(f"        ERROR: {result.error}")

    return result


# ============================================================
# AGGREGATE METRICS
# ============================================================

def aggregate(results: list[TestResult]) -> dict:
    """Compute aggregate metrics across all test results."""
    valid = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    if not valid:
        return {"error": "No valid results to aggregate"}

    def avg(vals): return sum(vals) / len(vals) if vals else 0.0

    # Per-category breakdown
    cats = defaultdict(list)
    for r in valid:
        cats[r.category].append(r)

    cat_summary = {}
    for cat, items in cats.items():
        cat_summary[cat] = {
            "count": len(items),
            "composite_score": avg([i.composite_score for i in items]),
            "tool_accuracy": avg([1.0 if i.router.tool_correct else 0.0 for i in items]),
            "domain_accuracy": avg([1.0 if i.router.domain_correct else 0.0 for i in items]),
            "recall_at_k": avg([i.retrieval.recall_at_k for i in items]),
            "precision_at_k": avg([i.retrieval.precision_at_k for i in items]),
            "mrr": avg([i.retrieval.mrr for i in items]),
            "keyword_overlap": avg([i.answer_metrics.keyword_overlap for i in items]),
            "completeness": avg([i.answer_metrics.completeness_score for i in items]),
            "citation_presence": avg([1.0 if i.citation.citations_present else 0.0 for i in items]),
            "avg_latency_ms": avg([i.latency.total_ms for i in items]),
        }

    # Global metrics
    pass_rate = avg([1.0 if r.composite_score >= 0.5 else 0.0 for r in valid])
    latencies = [r.latency.total_ms for r in valid if r.latency.total_ms > 0]

    # Find worst and best
    sorted_by_score = sorted(valid, key=lambda r: r.composite_score)
    weakest = [{"id": r.test_id, "category": r.category, "score": r.composite_score, "query": r.query[:80]} for r in sorted_by_score[:5]]
    strongest = [{"id": r.test_id, "category": r.category, "score": r.composite_score} for r in reversed(sorted_by_score[-5:])]

    # Failure examples
    failures = []
    for r in valid:
        if not r.router.tool_correct or r.composite_score < 0.3:
            failures.append({
                "id": r.test_id,
                "category": r.category,
                "query": r.query[:100],
                "expected_tool": r.expected_tool,
                "predicted_tool": r.predicted_tool,
                "score": r.composite_score,
                "error": r.error,
            })

    return {
        "run_timestamp": datetime.now().isoformat(),
        "total_cases": len(results),
        "valid_cases": len(valid),
        "error_cases": len(errors),
        "pass_rate": round(pass_rate, 4),
        "overall": {
            "composite_score": round(avg([r.composite_score for r in valid]), 4),
            "tool_accuracy": round(avg([1.0 if r.router.tool_correct else 0.0 for r in valid]), 4),
            "domain_accuracy": round(avg([1.0 if r.router.domain_correct else 0.0 for r in valid]), 4),
            "recall_at_k": round(avg([r.retrieval.recall_at_k for r in valid]), 4),
            "precision_at_k": round(avg([r.retrieval.precision_at_k for r in valid]), 4),
            "mrr": round(avg([r.retrieval.mrr for r in valid]), 4),
            "keyword_coverage": round(avg([r.retrieval.keyword_coverage for r in valid]), 4),
            "keyword_overlap": round(avg([r.answer_metrics.keyword_overlap for r in valid]), 4),
            "completeness_score": round(avg([r.answer_metrics.completeness_score for r in valid]), 4),
            "structure_adherence": round(avg([r.answer_metrics.structure_adherence for r in valid]), 4),
            "citation_presence": round(avg([1.0 if r.citation.citations_present else 0.0 for r in valid]), 4),
            "citation_field_completeness": round(avg([r.citation.avg_field_completeness for r in valid]), 4),
            "source_hint_match": round(avg([r.citation.source_hint_match for r in valid]), 4),
            "planner_confidence": round(avg([r.router.planner_confidence for r in valid]), 4),
            "planner_fallback_rate": round(avg([1.0 if r.router.planner_fallback else 0.0 for r in valid]), 4),
        },
        "latency": {
            "mean_total_ms": round(avg(latencies), 1),
            "p50_ms": round(sorted(latencies)[len(latencies)//2], 1) if latencies else 0,
            "p90_ms": round(sorted(latencies)[int(len(latencies)*0.9)], 1) if latencies else 0,
            "max_ms": round(max(latencies), 1) if latencies else 0,
            "min_ms": round(min(latencies), 1) if latencies else 0,
            "mean_planner_ms": round(avg([r.latency.planner_ms for r in valid if r.latency.planner_ms > 0]), 1),
            "mean_retrieval_ms": round(avg([r.latency.retrieval_ms for r in valid if r.latency.retrieval_ms > 0]), 1),
            "mean_rerank_ms": round(avg([r.latency.rerank_ms for r in valid if r.latency.rerank_ms > 0]), 1),
            "mean_generation_ms": round(avg([r.latency.generation_ms for r in valid if r.latency.generation_ms > 0]), 1),
        },
        "per_category": cat_summary,
        "weakest_queries": weakest,
        "strongest_queries": strongest,
        "failure_examples": failures[:10],
        "error_cases_detail": [{"id": r.test_id, "error": r.error} for r in errors],
    }


# ============================================================
# REPORTERS
# ============================================================

def write_jsonl(results: list[TestResult], path: Path) -> None:
    """Write per-case results as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            row = {
                "id": r.test_id,
                "category": r.category,
                "query": r.query,
                "expected_tool": r.expected_tool,
                "predicted_tool": r.predicted_tool,
                "expected_domain": r.expected_domain,
                "predicted_domain": r.predicted_domain,
                "composite_score": r.composite_score,
                "tool_correct": r.router.tool_correct,
                "domain_correct": r.router.domain_correct,
                "planner_confidence": r.router.planner_confidence,
                "planner_fallback": r.router.planner_fallback,
                "recall_at_k": r.retrieval.recall_at_k,
                "precision_at_k": r.retrieval.precision_at_k,
                "mrr": r.retrieval.mrr,
                "keyword_coverage": r.retrieval.keyword_coverage,
                "keyword_overlap": r.answer_metrics.keyword_overlap,
                "completeness_score": r.answer_metrics.completeness_score,
                "structure_adherence": r.answer_metrics.structure_adherence,
                "answer_length": r.answer_metrics.answer_length,
                "citation_count": r.citation.citation_count,
                "citation_field_completeness": r.citation.avg_field_completeness,
                "source_hint_match": r.citation.source_hint_match,
                "latency_total_ms": r.latency.total_ms,
                "latency_planner_ms": r.latency.planner_ms,
                "latency_retrieval_ms": r.latency.retrieval_ms,
                "latency_rerank_ms": r.latency.rerank_ms,
                "latency_generation_ms": r.latency.generation_ms,
                "answer_preview": r.answer[:200] if r.answer else "",
                "error": r.error,
                "timestamp": r.timestamp,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(results: list[TestResult], agg: dict, path: Path) -> None:
    """Write CSV summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "category", "query_short", "expected_tool", "predicted_tool", "tool_correct",
        "expected_domain", "predicted_domain", "domain_correct",
        "composite_score", "recall_at_k", "precision_at_k", "mrr", "keyword_overlap",
        "completeness_score", "structure_adherence", "citation_count",
        "latency_total_ms", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "id": r.test_id,
                "category": r.category,
                "query_short": r.query[:60],
                "expected_tool": r.expected_tool,
                "predicted_tool": r.predicted_tool,
                "tool_correct": r.router.tool_correct,
                "expected_domain": r.expected_domain,
                "predicted_domain": r.predicted_domain,
                "domain_correct": r.router.domain_correct,
                "composite_score": r.composite_score,
                "recall_at_k": r.retrieval.recall_at_k,
                "precision_at_k": r.retrieval.precision_at_k,
                "mrr": r.retrieval.mrr,
                "keyword_overlap": r.answer_metrics.keyword_overlap,
                "completeness_score": r.answer_metrics.completeness_score,
                "structure_adherence": r.answer_metrics.structure_adherence,
                "citation_count": r.citation.citation_count,
                "latency_total_ms": r.latency.total_ms,
                "error": r.error or "",
            })


def write_markdown_report(results: list[TestResult], agg: dict, path: Path, k: int) -> None:
    """Generate a comprehensive Markdown evaluation report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    o = agg.get("overall", {})
    lat = agg.get("latency", {})
    cats = agg.get("per_category", {})

    lines = []
    lines.append(f"# Industrial Energy Efficiency Copilot — Evaluation Report\n")
    lines.append(f"**Generated:** {agg.get('run_timestamp', 'N/A')}  \n")
    lines.append(f"**Total cases:** {agg['total_cases']} &nbsp;|&nbsp; **Valid:** {agg['valid_cases']} &nbsp;|&nbsp; **Errors:** {agg['error_cases']}\n")
    lines.append(f"**Pass rate (score ≥ 0.5):** {agg['pass_rate'] * 100:.1f}%\n\n")

    lines.append("---\n\n## 📊 Overall Metrics\n")
    lines.append("| Metric | Value | Threshold | Status |\n|--------|-------|-----------|--------|\n")

    thresholds = {
        "composite_score": ("≥ 0.60", 0.60),
        "tool_accuracy": ("≥ 0.80", 0.80),
        "domain_accuracy": ("≥ 0.85", 0.85),
        "recall_at_k": (f"≥ 0.60 @{k}", 0.60),
        "precision_at_k": (f"≥ 0.50 @{k}", 0.50),
        "mrr": ("≥ 0.65", 0.65),
        "keyword_coverage": ("≥ 0.60", 0.60),
        "keyword_overlap": ("≥ 0.50", 0.50),
        "completeness_score": ("≥ 0.45", 0.45),
        "citation_presence": ("≥ 0.80", 0.80),
    }

    metric_labels = {
        "composite_score": "Composite Score",
        "tool_accuracy": "Tool Accuracy",
        "domain_accuracy": "Domain Accuracy",
        f"recall_at_k": f"Recall@{k}",
        "precision_at_k": f"Precision@{k}",
        "mrr": "MRR",
        "keyword_coverage": "Keyword Coverage",
        "keyword_overlap": "Answer Keyword Overlap",
        "completeness_score": "Completeness Score",
        "citation_presence": "Citation Presence",
        "structure_adherence": "Structure Adherence",
        "planner_confidence": "Planner Confidence",
        "planner_fallback_rate": "Planner Fallback Rate",
    }

    for key, label in metric_labels.items():
        value = o.get(key, 0.0)
        if key in thresholds:
            th_label, th_val = thresholds[key]
            status = "✅ PASS" if value >= th_val else "❌ FAIL"
        else:
            th_label = "—"
            status = "ℹ️"
        lines.append(f"| {label} | `{value:.4f}` | {th_label} | {status} |\n")

    lines.append("\n---\n\n## ⏱ Latency Summary\n")
    lines.append("| Stage | Mean (ms) |\n|-------|----------|\n")
    lines.append(f"| Planner | `{lat.get('mean_planner_ms', 0):.0f}` |\n")
    lines.append(f"| Retrieval (dense+sparse+merge) | `{lat.get('mean_retrieval_ms', 0):.0f}` |\n")
    lines.append(f"| Reranking | `{lat.get('mean_rerank_ms', 0):.0f}` |\n")
    lines.append(f"| Answer Generation | `{lat.get('mean_generation_ms', 0):.0f}` |\n")
    lines.append(f"| **Total (mean)** | **`{lat.get('mean_total_ms', 0):.0f}`** |\n")
    lines.append(f"| Total (p50) | `{lat.get('p50_ms', 0):.0f}` |\n")
    lines.append(f"| Total (p90) | `{lat.get('p90_ms', 0):.0f}` |\n")
    lines.append(f"| Total (max) | `{lat.get('max_ms', 0):.0f}` |\n")

    lines.append("\n---\n\n## 🗂 Per-Category Performance\n")
    lines.append("| Category | Count | Composite | Tool Acc | Domain Acc | Recall@K | MRR | KW Overlap | Latency (ms) |\n")
    lines.append("|----------|-------|-----------|----------|------------|----------|-----|------------|-------------|\n")
    for cat, m in sorted(cats.items()):
        lines.append(
            f"| {cat} | {m['count']} "
            f"| `{m['composite_score']:.3f}` "
            f"| `{m['tool_accuracy']:.2f}` "
            f"| `{m['domain_accuracy']:.2f}` "
            f"| `{m['recall_at_k']:.3f}` "
            f"| `{m['mrr']:.3f}` "
            f"| `{m['keyword_overlap']:.3f}` "
            f"| `{m['avg_latency_ms']:.0f}` |\n"
        )

    lines.append("\n---\n\n## ❌ Failure Examples\n")
    failures = agg.get("failure_examples", [])
    if failures:
        for f in failures[:5]:
            lines.append(f"**{f['id']}** `[{f['category']}]` — score: `{f['score']:.3f}`  \n")
            lines.append(f"*Query:* {f['query']}  \n")
            lines.append(f"*Expected tool:* `{f['expected_tool']}` → *Got:* `{f['predicted_tool']}`\n\n")
    else:
        lines.append("*No significant failures.*\n")

    lines.append("\n---\n\n## 🔻 Weakest Queries\n")
    for w in agg.get("weakest_queries", []):
        lines.append(f"- `[score={w['score']:.3f}]` **{w['id']}** ({w['category']}): {w['query']}\n")

    lines.append("\n---\n\n## ✅ Strongest Queries\n")
    for s in agg.get("strongest_queries", []):
        lines.append(f"- `[score={s['score']:.3f}]` **{s['id']}** ({s['category']})\n")

    lines.append("\n---\n\n## 📋 Interpretation Guide\n")
    lines.append("| Score | Meaning |\n|-------|---------|\n")
    lines.append("| ≥ 0.75 | Excellent — production ready |\n")
    lines.append("| 0.60–0.75 | Good — minor tuning needed |\n")
    lines.append("| 0.45–0.60 | Fair — prompts or retrieval need work |\n")
    lines.append("| < 0.45 | Poor — systematic issue; re-run ingestion or check Ollama |\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def print_console_summary(agg: dict) -> None:
    """Print a concise console summary."""
    o = agg.get("overall", {})
    lat = agg.get("latency", {})
    cats = agg.get("per_category", {})

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY — Industrial Energy Efficiency Copilot")
    print("=" * 70)
    print(f"  Cases:          {agg['valid_cases']}/{agg['total_cases']} valid  ({agg['error_cases']} errors)")
    print(f"  Pass rate:      {agg['pass_rate'] * 100:.1f}%  (score ≥ 0.50)")
    print(f"\n  OVERALL SCORES:")
    print(f"  ┌─ Composite Score:    {o.get('composite_score', 0):.4f}")
    print(f"  ├─ Tool Accuracy:      {o.get('tool_accuracy', 0):.4f}  ({o.get('tool_accuracy', 0)*100:.1f}%)")
    print(f"  ├─ Domain Accuracy:    {o.get('domain_accuracy', 0):.4f}  ({o.get('domain_accuracy', 0)*100:.1f}%)")
    print(f"  ├─ Recall@K:          {o.get('recall_at_k', 0):.4f}")
    print(f"  ├─ Precision@K:       {o.get('precision_at_k', 0):.4f}")
    print(f"  ├─ MRR:               {o.get('mrr', 0):.4f}")
    print(f"  ├─ Keyword Overlap:   {o.get('keyword_overlap', 0):.4f}")
    print(f"  ├─ Completeness:      {o.get('completeness_score', 0):.4f}")
    print(f"  └─ Citation Rate:     {o.get('citation_presence', 0):.4f}")
    print(f"\n  LATENCY:")
    print(f"  ┌─ Mean total:        {lat.get('mean_total_ms', 0):.0f} ms")
    print(f"  ├─ p90:               {lat.get('p90_ms', 0):.0f} ms")
    print(f"  ├─ Planner:           {lat.get('mean_planner_ms', 0):.0f} ms")
    print(f"  └─ Generation:        {lat.get('mean_generation_ms', 0):.0f} ms")
    print(f"\n  PER-CATEGORY (composite score):")
    for cat, m in sorted(cats.items()):
        bar_n = int(m['composite_score'] * 20)
        bar = "█" * bar_n + "░" * (20 - bar_n)
        toolacc = f"{m['tool_accuracy']*100:.0f}%"
        print(f"  {cat:14s} [{bar}] {m['composite_score']:.3f}  tool={toolacc}")
    print("=" * 70)

    weakest = agg.get("weakest_queries", [])[:3]
    if weakest:
        print("  ⚠ Weakest queries:")
        for w in weakest:
            print(f"    [{w['score']:.3f}] {w['id']}: {w['query'][:60]}...")
    print("=" * 70 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Industrial Energy Efficiency Copilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument("--dataset", default="evaluation/synthetic_dataset.json", help="Path to evaluation dataset JSON")
    parser.add_argument("--outdir", default="evaluation/results", help="Output directory for reports")
    parser.add_argument("--k", type=int, default=5, help="K for Recall@K and Precision@K")
    parser.add_argument("--category", default=None, help="Only evaluate a specific category")
    parser.add_argument("--timeout", type=float, default=120.0, help="API timeout in seconds")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case output")
    parser.add_argument("--run-name", default=None, help="Name for this evaluation run (used in output filenames)")
    args = parser.parse_args()

    # ---- Load dataset ----
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)

    test_cases = [TestCase(**tc) for tc in raw]

    if args.category:
        test_cases = [tc for tc in test_cases if tc.category == args.category]
        if not test_cases:
            print(f"❌ No test cases for category '{args.category}'")
            sys.exit(1)

    # ---- Check backend health ----
    print(f"\n🔌 Checking backend: {args.base_url}")
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{args.base_url}/api/health")
            health = resp.json()
        if not health.get("index_loaded"):
            print("⚠️  WARNING: Backend reports indexes not loaded. Run ingestion first.")
            print("   Continuing anyway — results may be empty.\n")
        else:
            print(f"✅ Backend ready: {health.get('chunk_count', 0)} chunks indexed\n")
    except Exception as e:
        print(f"❌ Cannot reach backend: {e}")
        print("   Start it with: ./run_backend.sh")
        sys.exit(1)

    # ---- Run evaluation ----
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"📋 Evaluating {len(test_cases)} test cases (k={args.k})")
    print(f"📁 Output: {outdir}\n")

    results = []
    for i, tc in enumerate(test_cases, 1):
        if not args.quiet:
            print(f"[{i:2d}/{len(test_cases)}]", end="")
        result = evaluate_one(tc, args.base_url, args.k, verbose=not args.quiet)
        results.append(result)

    # ---- Aggregate ----
    agg = aggregate(results)

    # ---- Write outputs ----
    jsonl_path = outdir / "results.jsonl"
    csv_path = outdir / "results.csv"
    report_path = outdir / "report.md"
    agg_path = outdir / "aggregate.json"

    write_jsonl(results, jsonl_path)
    write_csv(results, agg, csv_path)
    write_markdown_report(results, agg, report_path, args.k)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    print_console_summary(agg)

    print(f"📁 Results saved to: {outdir}")
    print(f"   📄 {jsonl_path.name}   — per-case JSONL")
    print(f"   📊 {csv_path.name}    — CSV summary")
    print(f"   📝 {report_path.name}    — Markdown report")
    print(f"   📈 {agg_path.name} — aggregate JSON\n")


if __name__ == "__main__":
    main()
