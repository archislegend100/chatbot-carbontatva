import pytest
from app.routing.query_router import query_router
from app.ingestion.ocr_cleaner import ocr_cleaner

def test_query_router_formula():
    profile = query_router.route_query("What is the formula for motor efficiency?")
    assert profile.intent == "formula"
    assert profile.utility_domain == "electrical"
    assert profile.needs_formula is True

def test_query_router_troubleshoot():
    profile = query_router.route_query("Why is my boiler efficiency dropping?")
    assert profile.intent == "troubleshoot"
    assert profile.utility_domain == "thermal"
    assert profile.difficulty == "hard"
    assert profile.needs_parent_expansion is True

def test_ocr_cleaner():
    raw_text = "The boller efficlency is 85%."
    cleaned = ocr_cleaner.clean_text(raw_text)
    assert cleaned == "The boiler efficiency is 85%."
