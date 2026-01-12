"""
CLI tools for methylation HMM evaluation.

Provides command-line interfaces for:
- run_evaluation.py: Evaluate individual classifier configurations
- run_all_evaluations.py: Run all 8+ model configurations
"""

from pathlib import Path

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_SIGNAL_CSV = PROJECT_ROOT / "output" / "rep1" / "signal_at_cytosines_3way.csv"
DEFAULT_FULL_SIGNAL_CSV = PROJECT_ROOT / "output" / "rep1" / "signal_full_sequence.csv"
DEFAULT_BED_FILE = PROJECT_ROOT / "nanopore_ref_data" / "all_5mers_C_sites.bed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"
DEFAULT_KMER_MODEL = PROJECT_ROOT / "nanopore_ref_data" / "kmer_models" / "9mer_levels_v1.txt"
DEFAULT_REFERENCE = PROJECT_ROOT / "nanopore_ref_data" / "all_5mers.fa"
