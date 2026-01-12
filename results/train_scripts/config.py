"""Configuration for methylation HMM training scripts.

Defines paths, constants, and site type mappings.
"""

from pathlib import Path

# Project root (relative to this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
SIGNAL_CSV = PROJECT_ROOT / "output" / "rep1" / "signal_at_cytosines_3way.csv"
BED_FILE = PROJECT_ROOT / "nanopore_ref_data" / "all_5mers_C_sites.bed"
TRAINING_CSV = PROJECT_ROOT / "output" / "hmm_training_sequences.csv"  # Pre-pivoted (if exists)

# Output paths
MODEL_OUTPUT_DIR = PROJECT_ROOT / "results" / "models"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"

# Cytosine positions in the reference construct (0-indexed in reference)
POSITIONS = [38, 50, 62, 74, 86, 98, 110, 122]
POSITION_COLS = [str(p) for p in POSITIONS]  # Column names in training CSV

# Adapter/sequence names (32 synthetic constructs)
ADAPTERS = [f"5mers_rand_ref_adapter_{i:02d}" for i in range(1, 33)]
DEFAULT_ADAPTER = ADAPTERS[0]  # adapter_01 is the default

# Site type definitions from BED score column
SITE_TYPES = {
    0: "non_cpg",       # C not followed by G (cleanest signal)
    1: "cpg",           # CpG dinucleotide (biological target)
    2: "homopolymer",   # CC run (noisier)
}
SITE_TYPE_NAMES = {v: k for k, v in SITE_TYPES.items()}  # Reverse mapping

# Sample names in signal CSV
SAMPLE_CONTROL = "control"
SAMPLE_5MC = "5mC"
SAMPLE_5HMC = "5hmC"

# Classification mode definitions
BINARY_SAMPLES = [SAMPLE_CONTROL, SAMPLE_5MC]
THREE_WAY_SAMPLES = [SAMPLE_CONTROL, SAMPLE_5MC, SAMPLE_5HMC]

# Signal CSV column names
SIGNAL_COLS = {
    "sample": "sample",
    "chrom": "chrom",
    "position": "position",
    "read_id": "read_id",
    "mean_current": "mean_current",
    "std_current": "std_current",
    "dwell_time": "dwell_time",
    "n_samples": "n_samples",
}

# BED file column indices
BED_COLS = {
    "chrom": 0,
    "start": 1,
    "end": 2,
    "name": 3,
    "score": 4,  # Site type (0, 1, 2)
    "strand": 5,
}
