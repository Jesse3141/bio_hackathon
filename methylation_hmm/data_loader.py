"""
Data loading and preprocessing for Profile HMM.

Handles TSV parsing, z-score normalization, and batching.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SegmentedRead:
    """A single nanopore read with segmented current values."""
    read_id: str
    segments: np.ndarray      # Array of mean currents (z-scored)
    segment_stds: np.ndarray  # Array of segment std devs
    label: Optional[str] = None  # 'control' or '5mC' if known

    def __len__(self) -> int:
        return len(self.segments)


class DataLoader:
    """Load and preprocess nanopore segment data."""

    def __init__(
        self,
        min_segments: int = 50,
        max_segments: int = 500
    ):
        """
        Initialize data loader.

        Args:
            min_segments: Minimum segments per read to keep
            max_segments: Maximum segments per read to keep
        """
        self.min_segments = min_segments
        self.max_segments = max_segments

    def load_tsv(self, filepath: str) -> pd.DataFrame:
        """
        Load TSV with columns (read_id, segment_idx, mean_current, std).

        Args:
            filepath: Path to TSV file

        Returns:
            DataFrame with segment data
        """
        df = pd.read_csv(
            filepath,
            sep='\t',
            names=['read_id', 'segment_idx', 'mean_current', 'std'],
            header=None
        )

        # Handle case where file has header row
        if df.iloc[0]['read_id'] == 'read_id':
            df = df.iloc[1:].reset_index(drop=True)
            df['segment_idx'] = df['segment_idx'].astype(int)
            df['mean_current'] = df['mean_current'].astype(float)
            df['std'] = df['std'].astype(float)

        print(f"Loaded {len(df)} segments from {len(df['read_id'].unique())} reads")
        return df

    def group_by_read(
        self,
        df: pd.DataFrame,
        label: Optional[str] = None
    ) -> List[SegmentedRead]:
        """
        Group segments into reads.

        Args:
            df: DataFrame from load_tsv
            label: Optional label for all reads ('control' or '5mC')

        Returns:
            List of SegmentedRead objects
        """
        reads = []

        for read_id, group in df.groupby('read_id'):
            # Sort by segment index
            group = group.sort_values('segment_idx')

            segments = group['mean_current'].values.astype(np.float32)
            stds = group['std'].values.astype(np.float32)

            reads.append(SegmentedRead(
                read_id=str(read_id),
                segments=segments,
                segment_stds=stds,
                label=label
            ))

        return reads

    def filter_reads(
        self,
        reads: List[SegmentedRead]
    ) -> List[SegmentedRead]:
        """
        Filter reads by segment count.

        Args:
            reads: List of SegmentedRead objects

        Returns:
            Filtered list
        """
        filtered = [
            r for r in reads
            if self.min_segments <= len(r) <= self.max_segments
        ]

        print(f"Filtered: {len(filtered)}/{len(reads)} reads "
              f"(kept {self.min_segments}-{self.max_segments} segments)")

        return filtered

    def normalize_to_zscore(
        self,
        reads: List[SegmentedRead]
    ) -> List[SegmentedRead]:
        """
        Z-score normalize current values per-read.

        Converts raw pA values to z-scores using per-read statistics.
        This makes the data compatible with the 9mer model.

        Args:
            reads: List of SegmentedRead objects

        Returns:
            New list with z-scored segments
        """
        normalized = []

        for read in reads:
            mean = read.segments.mean()
            std = read.segments.std()

            if std < 1e-6:  # Avoid division by zero
                std = 1.0

            z_segments = (read.segments - mean) / std

            normalized.append(SegmentedRead(
                read_id=read.read_id,
                segments=z_segments,
                segment_stds=read.segment_stds / std,  # Scale stds too
                label=read.label
            ))

        return normalized

    def load_and_preprocess(
        self,
        filepath: str,
        label: Optional[str] = None,
        normalize: bool = True
    ) -> List[SegmentedRead]:
        """
        Full pipeline: load, group, filter, normalize.

        Args:
            filepath: Path to TSV file
            label: Optional label for reads
            normalize: Whether to z-score normalize

        Returns:
            List of preprocessed SegmentedRead objects
        """
        df = self.load_tsv(filepath)
        reads = self.group_by_read(df, label)
        reads = self.filter_reads(reads)

        if normalize:
            reads = self.normalize_to_zscore(reads)

        return reads

    def prepare_for_hmm(
        self,
        reads: List[SegmentedRead],
        max_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert reads to padded arrays for batch processing.

        Args:
            reads: List of SegmentedRead objects
            max_length: Max sequence length (None = use longest)

        Returns:
            (sequences, lengths, read_ids)
            - sequences: (n_reads, max_length) padded array
            - lengths: (n_reads,) actual lengths
            - read_ids: list of read IDs
        """
        if max_length is None:
            max_length = max(len(r) for r in reads)

        n_reads = len(reads)
        sequences = np.zeros((n_reads, max_length), dtype=np.float32)
        lengths = np.zeros(n_reads, dtype=np.int32)
        read_ids = []

        for i, read in enumerate(reads):
            length = min(len(read), max_length)
            sequences[i, :length] = read.segments[:length]
            lengths[i] = length
            read_ids.append(read.read_id)

        return sequences, lengths, read_ids


def load_reference_sequence(filepath: str) -> str:
    """
    Load reference sequence from FASTA file.

    Args:
        filepath: Path to FASTA file

    Returns:
        DNA sequence string (uppercase)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip header line(s), concatenate sequence lines
    sequence = ''
    for line in lines:
        line = line.strip()
        if not line.startswith('>'):
            sequence += line

    return sequence.upper()
