"""Training scripts for methylation HMM classification.

This module provides data loading and preprocessing utilities for training
HMM-based methylation classifiers. Supports both binary (C vs 5mC) and
3-way (C vs 5mC vs 5hmC) classification tasks.

Main functions:
    load_bed_with_site_types: Load BED file with cytosine context scores
    load_signal_data: Load raw signal measurements
    pivot_to_training_format: Convert to wide format for HMM training
    prepare_training_data: High-level function for full data preparation

Example usage:
    from results.train_scripts import prepare_training_data

    # Binary classification on adapter_01
    df = prepare_training_data(
        adapter="5mers_rand_ref_adapter_01",
        classification="binary"
    )

    # 3-way classification on all adapters
    df = prepare_training_data(classification="3way")
"""

from .data_loader import (
    load_bed_with_site_types,
    load_signal_data,
    pivot_to_training_format,
    add_site_types,
    get_positions_by_site_type,
    prepare_training_data,
    get_site_type_summary,
    get_data_summary,
)

from .config import (
    POSITIONS,
    POSITION_COLS,
    ADAPTERS,
    DEFAULT_ADAPTER,
    SITE_TYPES,
    BINARY_SAMPLES,
    THREE_WAY_SAMPLES,
    SIGNAL_CSV,
    BED_FILE,
)

__all__ = [
    # Data loading
    "load_bed_with_site_types",
    "load_signal_data",
    "pivot_to_training_format",
    "add_site_types",
    "get_positions_by_site_type",
    "prepare_training_data",
    "get_site_type_summary",
    "get_data_summary",
    # Config
    "POSITIONS",
    "POSITION_COLS",
    "ADAPTERS",
    "DEFAULT_ADAPTER",
    "SITE_TYPES",
    "BINARY_SAMPLES",
    "THREE_WAY_SAMPLES",
    "SIGNAL_CSV",
    "BED_FILE",
]
