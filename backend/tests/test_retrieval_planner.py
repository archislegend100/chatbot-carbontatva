import pytest
from app.routing.retrieval_planner import retrieval_planner
from app.routing.query_router import QueryProfile
from app.models.chat import AdvancedRetrievalOptions
from app.config.settings import settings

@pytest.fixture
def base_profile():
    return QueryProfile(
        intent="qa",
        utility_domain="thermal",
        difficulty="easy",
        needs_table=False,
        needs_formula=False,
        needs_parent_expansion=False,
        needs_colbert=False,
        needs_hyde=False,
        needs_verification=False
    )

@pytest.fixture
def hard_profile():
    return QueryProfile(
        intent="troubleshoot",
        utility_domain="mixed",
        difficulty="hard",
        needs_table=True,
        needs_formula=True,
        needs_parent_expansion=True,
        needs_colbert=True,
        needs_hyde=True,
        needs_verification=True
    )

def test_default_mode_is_auto(base_profile):
    options = AdvancedRetrievalOptions()
    # Missing/invalid mode should default to auto
    plan = retrieval_planner.create_plan("what is boiler efficiency", base_profile, "invalid_mode", options)
    assert plan.use_colbert == False # Base profile doesn't need it
    assert plan.use_dense == True

def test_fast_mode_disables_heavy_ops(hard_profile):
    options = AdvancedRetrievalOptions(force_colbert=True, force_hyde=True)
    plan = retrieval_planner.create_plan("why is my boiler broken", hard_profile, "fast", options)
    # Even if profile requires it or force is True, fast mode overrides it
    assert plan.use_colbert == False
    assert plan.use_hyde == False
    assert plan.use_multi_query == False
    assert plan.top_k_dense == settings.FAST_TOP_K_DENSE

def test_deep_mode_allows_colbert(hard_profile):
    # Temporarily enable features for test
    settings.ENABLE_COLBERT = True
    options = AdvancedRetrievalOptions()
    plan = retrieval_planner.create_plan("compare VFD vs fluid coupling", hard_profile, "deep", options)
    assert plan.use_colbert == True
    assert plan.use_parent_expansion == True

def test_research_mode_returns_debug_and_allows_overrides(base_profile):
    settings.ENABLE_COLBERT = True
    options = AdvancedRetrievalOptions(force_colbert=True, show_chunks=True)
    plan = retrieval_planner.create_plan("basic question", base_profile, "research", options)
    assert plan.return_debug == True
    # Overrides apply in research
    assert plan.use_colbert == True

def test_force_hyde_ignored_if_disabled_globally(base_profile):
    settings.ENABLE_HYDE = False
    options = AdvancedRetrievalOptions(force_hyde=True)
    plan = retrieval_planner.create_plan("basic question", base_profile, "research", options)
    # Global disable overrides force option
    assert plan.use_hyde == False

def test_existing_simple_payload(base_profile):
    options = AdvancedRetrievalOptions()
    plan = retrieval_planner.create_plan("hello", base_profile, "auto", options)
    assert plan.use_dense == True
    assert plan.use_sparse == True
    assert plan.return_debug == False
