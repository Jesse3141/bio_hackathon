"""Data loading utilities for methylation HMM training.

Supports both binary (C vs 5mC) and 3-way (C vs 5mC vs 5hmC) classification.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    POSITIONS,
    POSITION_COLS,
    SITE_TYPES,
    SIGNAL_COLS,
    BED_COLS,
    BINARY_SAMPLES,
    THREE_WAY_SAMPLES,
    SIGNAL_CSV,
    BED_FILE,
    DEFAULT_ADAPTER,
)


def load_bed_with_site_types(bed_path: str | Path | None = None) -> pd.DataFrame:
    """Load BED file with site type scores.

    Args:
        bed_path: Path to BED file. Defaults to config BED_FILE.

    Returns:
        DataFrame with columns: chrom, position, site_type
        - chrom: adapter/sequence name (e.g., "5mers_rand_ref_adapter_01")
        - position: cytosine position (38, 50, 62, 74, 86, 98, 110, 122)
        - site_type: 0 (non-CpG), 1 (CpG), or 2 (homopolymer)
    """
    if bed_path is None:
        bed_path = BED_FILE

    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name", "site_type", "strand"],
    )

    # Use start position (0-indexed in BED)
    df["position"] = df["start"]

    return df[["chrom", "position", "site_type"]]


def load_signal_data(
    csv_path: str | Path | None = None,
    adapters: list[str] | None = None,
    samples: list[str] | None = None,
) -> pd.DataFrame:
    """Load raw signal CSV (long format).

    Args:
        csv_path: Path to signal_at_cytosines_3way.csv. Defaults to config SIGNAL_CSV.
        adapters: Filter to specific adapters (e.g., ["5mers_rand_ref_adapter_01"]).
                  None = all adapters.
        samples: Filter to specific samples. Options:
                  - ["control", "5mC"] for binary classification
                  - ["control", "5mC", "5hmC"] for 3-way classification
                  - None = all samples

    Returns:
        DataFrame with columns: sample, chrom, position, read_id, mean_current, ...
    """
    if csv_path is None:
        csv_path = SIGNAL_CSV

    df = pd.read_csv(csv_path)

    # Filter by adapters
    if adapters is not None:
        df = df[df["chrom"].isin(adapters)]

    # Filter by samples
    if samples is not None:
        df = df[df["sample"].isin(samples)]

    return df


def pivot_to_training_format(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format signal data to wide format for HMM training.

    Input: one row per (read, position)
    Output: one row per read with columns [sample, chrom, read_id, 38, 50, ...]

    Args:
        signal_df: Long-format signal DataFrame from load_signal_data()

    Returns:
        Wide-format DataFrame suitable for HMM training
    """
    # Pivot: index by (sample, chrom, read_id), columns by position, values = mean_current
    pivoted = signal_df.pivot_table(
        index=["sample", "chrom", "read_id"],
        columns="position",
        values="mean_current",
        aggfunc="first",  # One value per read-position
    ).reset_index()

    # Rename position columns to strings
    pivoted.columns = [
        str(col) if isinstance(col, int) else col for col in pivoted.columns
    ]

    return pivoted


def add_site_types(
    training_df: pd.DataFrame, bed_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add site_type columns for each position based on BED scores.

    Creates columns site_type_38, site_type_50, etc. for each position.

    Args:
        training_df: Wide-format training DataFrame
        bed_df: BED DataFrame from load_bed_with_site_types(). Loaded if None.

    Returns:
        DataFrame with additional site_type_XX columns
    """
    if bed_df is None:
        bed_df = load_bed_with_site_types()

    result = training_df.copy()

    # For each chrom (adapter), look up site types for each position
    for pos in POSITIONS:
        pos_str = str(pos)
        site_type_col = f"site_type_{pos}"

        # Create a lookup: chrom -> site_type at this position
        pos_bed = bed_df[bed_df["position"] == pos].set_index("chrom")["site_type"]

        # Map chrom to site_type
        result[site_type_col] = result["chrom"].map(pos_bed)

    return result


def get_positions_by_site_type(
    bed_df: pd.DataFrame | None = None, chrom: str | None = None
) -> dict[int, list[int]]:
    """Get positions grouped by site type for a given adapter.

    Args:
        bed_df: BED DataFrame from load_bed_with_site_types(). Loaded if None.
        chrom: Specific adapter to query. If None, aggregates across all adapters.

    Returns:
        Dictionary mapping site_type (0, 1, 2) to list of positions.
        Example: {0: [38, 50, 62, 110, 122], 1: [86, 98], 2: [74]}
    """
    if bed_df is None:
        bed_df = load_bed_with_site_types()

    if chrom is not None:
        bed_df = bed_df[bed_df["chrom"] == chrom]

    result: dict[int, list[int]] = {0: [], 1: [], 2: []}

    for site_type in [0, 1, 2]:
        positions = bed_df[bed_df["site_type"] == site_type]["position"].unique()
        result[site_type] = sorted(positions.tolist())

    return result


def prepare_training_data(
    signal_csv: str | Path | None = None,
    bed_path: str | Path | None = None,
    adapter: str | None = None,
    classification: str = "binary",
) -> pd.DataFrame:
    """High-level function to prepare training-ready data.

    Args:
        signal_csv: Path to signal CSV. Defaults to config.
        bed_path: Path to BED file. Defaults to config.
        adapter: Single adapter to use (e.g., "5mers_rand_ref_adapter_01").
                 None = all adapters.
        classification: "binary" (control vs 5mC) or "3way" (all 3 classes)

    Returns:
        Wide-format DataFrame ready for HMM training, with site_type columns
    """
    # Determine samples based on classification mode
    if classification == "binary":
        samples = BINARY_SAMPLES
    elif classification == "3way":
        samples = THREE_WAY_SAMPLES
    else:
        raise ValueError(f"Unknown classification mode: {classification}")

    # Filter adapters
    adapters = [adapter] if adapter else None

    # Load and process
    signal_df = load_signal_data(signal_csv, adapters=adapters, samples=samples)
    training_df = pivot_to_training_format(signal_df)

    # Add site types
    bed_df = load_bed_with_site_types(bed_path)
    training_df = add_site_types(training_df, bed_df)

    return training_df


def get_site_type_summary(bed_path: str | Path | None = None) -> pd.DataFrame:
    """Get summary of site types across all adapters.

    Returns:
        DataFrame with counts of each site type
    """
    bed_df = load_bed_with_site_types(bed_path)

    summary = (
        bed_df.groupby("site_type")
        .agg(
            count=("position", "count"),
            positions=("position", lambda x: sorted(x.unique().tolist())),
        )
        .reset_index()
    )

    summary["name"] = summary["site_type"].map(SITE_TYPES)

    return summary[["site_type", "name", "count", "positions"]]


def get_data_summary(
    signal_csv: str | Path | None = None,
    adapters: list[str] | None = None,
) -> pd.DataFrame:
    """Get summary of signal data.

    Returns:
        DataFrame with read counts per sample and adapter
    """
    df = load_signal_data(signal_csv, adapters=adapters)

    # Count unique reads per sample and chrom
    summary = (
        df.groupby(["sample", "chrom"])["read_id"]
        .nunique()
        .reset_index()
        .rename(columns={"read_id": "n_reads"})
    )

    return summary
