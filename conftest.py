"""Pytest configuration: mock optional dependencies not installed in CI.

dashboard.py imports streamlit at module level, but the functions under test
(_pred_grid_to_rgba, _get_wgs84_bounds) don't call any streamlit APIs, so a
MagicMock satisfies the import without needing streamlit installed.
"""
import sys
from unittest.mock import MagicMock

sys.modules.setdefault('streamlit', MagicMock())
