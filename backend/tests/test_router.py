"""
tests/test_router.py
====================
Tests for the query classifier and router.

Run: pytest backend/tests/test_router.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class TestQueryClassifier:
    """Test the query classifier."""

    @pytest.fixture
    def classifier(self):
        from app.router.query_classifier import QueryClassifier
        return QueryClassifier()  # No LLM client — rule-based only

    def test_qa_classification(self, classifier):
        from app.models.schemas import ToolMode
        result = classifier.classify("What is the optimum excess air for a coal boiler?")
        assert result.tool_mode in [ToolMode.QA, ToolMode.EXPLAINER]

    def test_thermal_domain_detection(self, classifier):
        from app.models.schemas import UtilityDomain
        result = classifier.classify("How does a steam trap work?")
        assert result.utility_domain == UtilityDomain.THERMAL

    def test_electrical_domain_detection(self, classifier):
        from app.models.schemas import UtilityDomain
        result = classifier.classify("How to improve power factor using capacitors?")
        assert result.utility_domain == UtilityDomain.ELECTRICAL

    def test_troubleshoot_detection(self, classifier):
        from app.models.schemas import ToolMode
        result = classifier.classify("Why is my boiler efficiency so low?")
        assert result.tool_mode in [ToolMode.TROUBLESHOOT, ToolMode.QA]

    def test_checklist_detection(self, classifier):
        from app.models.schemas import ToolMode
        result = classifier.classify("Generate an energy audit checklist for motors")
        assert result.tool_mode == ToolMode.CHECKLIST

    def test_comparison_detection(self, classifier):
        from app.models.schemas import ToolMode
        result = classifier.classify("Compare FBC boilers vs conventional boilers")
        assert result.tool_mode == ToolMode.COMPARISON

    def test_summarize_detection(self, classifier):
        from app.models.schemas import ToolMode
        result = classifier.classify("Summarize the chapter on waste heat recovery")
        assert result.tool_mode == ToolMode.SUMMARIZE

    def test_tag_extraction(self, classifier):
        result = classifier.classify("How to improve VFD motor efficiency?")
        assert "motor" in result.equipment_tags or "vfd" in result.equipment_tags

    def test_short_query_handled(self, classifier):
        # Should not crash on very short inputs
        result = classifier.classify("boilers?")
        assert result is not None

    def test_confidence_range(self, classifier):
        result = classifier.classify("What is cogeneration?")
        assert 0.0 <= result.confidence <= 1.0


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_query_request_validation(self):
        from app.models.schemas import QueryRequest, ToolMode
        req = QueryRequest(query="What is boiler efficiency?")
        assert req.tool_mode == ToolMode.AUTO

    def test_query_request_min_length(self):
        from app.models.schemas import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(query="ab")  # Too short

    def test_answer_response_structure(self):
        from app.models.schemas import AnswerResponse, ToolMode
        resp = AnswerResponse(
            query="test",
            tool_mode=ToolMode.QA,
            answer="Test answer",
        )
        assert resp.citations == []
        assert resp.follow_up_suggestions == []
