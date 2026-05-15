"""
conftest.py
===========
pytest configuration — adds project root to sys.path so
that 'from app.xxx import yyy' works from any test file.
"""

import sys
from pathlib import Path

# Project root = parent of this conftest.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
