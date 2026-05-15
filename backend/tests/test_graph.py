"""
backend/tests/test_graph.py
============================
Tests for the LangGraph orchestration layer.

Tests:
- CopilotState TypedDict structure
- planner.py (rule-based fallback, JSON parse, prompt building)
- graph.py (graph compilation, node ordering)

Does NOT require Ollama or ChromaDB to be running.
"""

from __future__ import annotations

import json
import pytest


class TestCopilotState:
    """Test the state TypedDict."""

    def test_state_can_be_created_empty(self):
        from app.graph.state import CopilotState
        state: CopilotState = {}
        assert isinstance(state, dict)

    def test_state_accepts_all_fields(self):
        from app.graph.state import CopilotState
        state: CopilotState = {
            "user_query": "What is boiler efficiency?",
            "session_id": "test-123",
            "explanation_level": "engineer",
            "normalized_query": "what is boiler efficiency",
            "debug_trace": ["normalize_query"],
            "latency": {"normalize_query": 1.2},
        }
        assert state["user_query"] == "What is boiler efficiency?"
        assert len(state["debug_trace"]) == 1


class TestPlannerRuleBased:
    """Test the rule-based fallback planner."""

    def test_qa_detection(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("What is boiler efficiency?")
        assert plan["selected_tool"] == "qa"

    def test_troubleshoot_detection(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("Why is my boiler efficiency dropping below 75%?")
        assert plan["selected_tool"] == "troubleshoot"

    def test_checklist_detection(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("Generate an audit checklist for boilers")
        assert plan["selected_tool"] == "checklist"

    def test_comparison_detection(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("Compare fire tube vs water tube boilers")
        assert plan["selected_tool"] == "comparison"

    def test_opportunity_detection(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("How to improve energy saving opportunities in motors?")
        assert plan["selected_tool"] == "opportunity"

    def test_thermal_domain_detection(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("What is boiler efficiency?")
        assert plan["utility_domain"] == "thermal"

    def test_electrical_domain_detection(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("How does power factor affect motor efficiency?")
        assert plan["utility_domain"] == "electrical"

    def test_domain_hint_overrides(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("What is efficiency?", domain_hint="thermal")
        assert plan["utility_domain"] == "thermal"

    def test_plan_has_required_keys(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("Explain the combustion process")
        required = ["query_type", "utility_domain", "equipment_tags",
                    "candidate_tools", "selected_tool", "confidence",
                    "why_selected", "retrieval_profile", "response_style"]
        for key in required:
            assert key in plan, f"Missing key: {key}"

    def test_equipment_tags_extraction(self):
        from app.graph.planner import _rule_based_plan
        plan = _rule_based_plan("Boiler steam trap efficiency for condensate recovery")
        assert len(plan["equipment_tags"]) > 0


class TestPlannerJsonParse:
    """Test JSON parsing logic."""

    def test_valid_json_parse(self):
        from app.graph.planner import parse_planner_response
        raw = json.dumps({
            "query_type": "qa",
            "utility_domain": "thermal",
            "equipment_tags": ["boiler"],
            "candidate_tools": ["qa", "explainer"],
            "selected_tool": "qa",
            "confidence": 0.9,
            "why_selected": "Direct factual question",
            "retrieval_profile": "semantic",
            "response_style": "direct",
        })
        result = parse_planner_response(raw)
        assert result is not None
        assert result["selected_tool"] == "qa"

    def test_json_embedded_in_text_parse(self):
        from app.graph.planner import parse_planner_response
        raw = 'Here is my routing decision:\n{"selected_tool": "checklist", "confidence": 0.8, "query_type": "checklist", "utility_domain": "thermal", "equipment_tags": [], "candidate_tools": ["checklist"], "why_selected": "audit", "retrieval_profile": "list", "response_style": "checklist"}'
        result = parse_planner_response(raw)
        assert result is not None
        assert result["selected_tool"] == "checklist"

    def test_invalid_json_returns_none(self):
        from app.graph.planner import parse_planner_response
        result = parse_planner_response("This is not JSON at all")
        assert result is None

    def test_empty_returns_none(self):
        from app.graph.planner import parse_planner_response
        result = parse_planner_response("")
        assert result is None


class TestGraphCompilation:
    """Test LangGraph graph compilation."""

    def test_graph_builds_without_error(self):
        """Graph should compile even with mock components."""
        from app.graph.graph import build_graph
        from unittest.mock import MagicMock, AsyncMock

        mock_llm = AsyncMock()
        mock_retriever = MagicMock()
        mock_retriever._rrf_fuse.return_value = []
        mock_retriever.dense_retriever.retrieve.return_value = []
        mock_retriever.sparse_retriever.retrieve.return_value = []
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        mock_vs = MagicMock()
        mock_vs.result_to_citation.return_value = {}

        graph = build_graph(mock_llm, mock_retriever, mock_reranker, mock_vs)
        assert graph is not None

    def test_normalize_query_node(self):
        from app.graph.nodes import normalize_query
        state = {"user_query": "  What IS boiler EFFICIENCY?  ", "debug_trace": [], "latency": {}}
        result = normalize_query(state)
        assert "normalized_query" in result
        assert result["normalized_query"] == "what is boiler efficiency?"
        assert "normalize_query" in result["debug_trace"]
