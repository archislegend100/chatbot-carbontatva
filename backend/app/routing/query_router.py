import re
from typing import List, Dict, Any
from pydantic import BaseModel

class QueryProfile(BaseModel):
    intent: str
    utility_domain: str
    difficulty: str
    needs_table: bool
    needs_formula: bool
    needs_parent_expansion: bool
    needs_colbert: bool
    needs_hyde: bool
    needs_verification: bool

class QueryRouter:
    def __init__(self):
        self.formula_keywords = ['formula', 'calculate', 'calculation', 'equation', '=', 'η', 'sec', 'efficiency']
        self.table_keywords = ['value', 'range', 'recommended', 'optimum', 'typical', 'table', 'compare', 'percentage']
        self.troubleshoot_keywords = ['why', 'causes', 'troubleshoot', 'issue', 'problem', 'low', 'high', 'drop']
        self.comparison_keywords = ['compare', 'difference', 'vs', 'better', 'instead']
        
        self.thermal_keywords = ['boiler', 'furnace', 'steam', 'condensate', 'insulation', 'heat', 'flue gas', 'whr']
        self.electrical_keywords = ['motor', 'pump', 'fan', 'compressor', 'vfd', 'lighting', 'transformer', 'power factor']

    def route_query(self, query: str) -> QueryProfile:
        q_lower = query.lower()
        
        # Domain classification
        is_thermal = any(k in q_lower for k in self.thermal_keywords)
        is_electrical = any(k in q_lower for k in self.electrical_keywords)
        if is_thermal and is_electrical:
            domain = "mixed"
        elif is_thermal:
            domain = "thermal"
        elif is_electrical:
            domain = "electrical"
        else:
            domain = "unknown"

        # Intent classification
        intent = "qa"
        if any(k in q_lower for k in self.formula_keywords):
            intent = "formula"
        elif any(k in q_lower for k in self.table_keywords):
            intent = "table_lookup"
        elif any(k in q_lower for k in self.troubleshoot_keywords):
            intent = "troubleshoot"
        elif any(k in q_lower for k in self.comparison_keywords):
            intent = "comparison"
        elif "checklist" in q_lower:
            intent = "checklist"
        elif "summary" in q_lower or "summarize" in q_lower:
            intent = "summarize"
        elif "explain" in q_lower or "what is" in q_lower:
            intent = "explainer"

        # Difficulty classification
        is_hard = intent in ["troubleshoot", "comparison", "formula", "summarize"] or len(q_lower.split()) > 10
        difficulty = "hard" if is_hard else "easy"

        # Retrieval Needs
        needs_table = intent == "table_lookup" or any(k in q_lower for k in self.table_keywords)
        needs_formula = intent == "formula" or any(k in q_lower for k in self.formula_keywords)
        needs_parent_expansion = intent in ["explainer", "troubleshoot", "comparison", "formula", "summarize"]
        needs_colbert = is_hard or needs_table or needs_formula
        needs_hyde = intent in ["explainer", "troubleshoot"] and not needs_formula and not needs_table
        needs_verification = intent in ["formula", "troubleshoot", "comparison"] or needs_table

        return QueryProfile(
            intent=intent,
            utility_domain=domain,
            difficulty=difficulty,
            needs_table=needs_table,
            needs_formula=needs_formula,
            needs_parent_expansion=needs_parent_expansion,
            needs_colbert=needs_colbert,
            needs_hyde=needs_hyde,
            needs_verification=needs_verification
        )

query_router = QueryRouter()
