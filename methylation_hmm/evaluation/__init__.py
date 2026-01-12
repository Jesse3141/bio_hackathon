"""
Unified evaluation framework for methylation HMM classifiers.

This module provides comprehensive evaluation tools including:
- UnifiedMetrics: Standard metrics dataclass for all classifiers
- UnifiedEvaluator: Evaluate any classifier with site-type breakdown
- Output formatters for JSON, CSV, and plots
"""

from .framework import UnifiedMetrics, UnifiedEvaluator
from .site_type_metrics import compute_accuracy_by_site_type, get_site_type_positions
from .output_formatters import save_json, save_csv, generate_plots

__all__ = [
    "UnifiedMetrics",
    "UnifiedEvaluator",
    "compute_accuracy_by_site_type",
    "get_site_type_positions",
    "save_json",
    "save_csv",
    "generate_plots",
]
