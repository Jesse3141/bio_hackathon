"""
Configuration for the Profile HMM.
"""

from dataclasses import dataclass, field
from typing import Tuple
from pathlib import Path


@dataclass
class HMMConfig:
    """Configuration for Profile HMM construction and training."""

    # Reference sequence (will be loaded from file if not provided)
    reference_sequence: str = ""
    sequence_length: int = 155

    # Cytosine positions for binary forks (0-indexed)
    cytosine_positions: Tuple[int, ...] = (38, 50, 62, 74, 86, 98, 110, 122)

    # Paths to data files
    kmer_model_path: str = ""
    reference_fasta_path: str = ""

    # State transition probabilities
    p_match: float = 0.90       # Entry -> Match
    p_insert: float = 0.05      # Entry -> Insert
    p_delete: float = 0.05      # Entry -> Skip (next position)
    p_match_self_loop: float = 0.10   # Match -> Match (oversegmentation)
    p_insert_self_loop: float = 0.30  # Insert -> Insert

    # Fork priors (equal by default)
    p_canonical: float = 0.50   # Prior P(C)
    p_methylated: float = 0.50  # Prior P(5mC)

    # Emission parameters
    methylation_shift: float = 0.8    # Z-score shift for 5mC vs C
    methylation_std_factor: float = 1.2  # Std multiplier for 5mC
    insert_std: float = 5.0           # Wide std for insert state

    # Training parameters
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    filter_score_threshold: float = 0.10  # Keep top ~26% reads

    # Data filtering
    min_segments_per_read: int = 50
    max_segments_per_read: int = 500

    def __post_init__(self):
        """Validate configuration."""
        assert self.p_match + self.p_insert + self.p_delete == 1.0, \
            "Transition probs must sum to 1.0"
        assert self.p_canonical + self.p_methylated == 1.0, \
            "Fork priors must sum to 1.0"
        assert len(self.cytosine_positions) == 8, \
            "Expected 8 cytosine positions"


def default_config() -> HMMConfig:
    """Create config with default paths for this project."""
    base_path = Path(__file__).parent.parent

    return HMMConfig(
        kmer_model_path=str(base_path / "nanopore_ref_data" / "kmer_models" / "9mer_levels_v1.txt"),
        reference_fasta_path=str(base_path / "filtered_pod_files" / "adapter_1_seq"),
    )
