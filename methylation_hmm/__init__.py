"""
Profile HMM for Nanopore Methylation Detection

Classifies cytosines as C or 5mC using a simplified Profile HMM
with binary forks at cytosine positions.

Usage:
    from methylation_hmm import default_config, build_hmm_from_config
    from methylation_hmm import MethylationClassifier, BaumWelchTrainer

    # Build model
    config = default_config()
    model, builder = build_hmm_from_config(config)

    # Classify reads
    classifier = MethylationClassifier(model, builder, config)
    results = classifier.classify_batch(reads)
"""

from .config import HMMConfig, default_config
from .kmer_model import KmerModel
from .data_loader import DataLoader, SegmentedRead, load_reference_sequence
from .distributions import DistributionFactory, StateInfo
from .hmm_builder import ProfileHMMBuilder, build_hmm_from_config
from .classification import MethylationClassifier, ClassificationResult, ReadClassification
from .training import BaumWelchTrainer, train_from_tsvs
from .evaluation import Evaluator, EvaluationMetrics, load_ground_truth_labels

__version__ = "0.1.0"

__all__ = [
    # Config
    "HMMConfig",
    "default_config",
    # Data
    "KmerModel",
    "DataLoader",
    "SegmentedRead",
    "load_reference_sequence",
    # Model building
    "DistributionFactory",
    "StateInfo",
    "ProfileHMMBuilder",
    "build_hmm_from_config",
    # Classification
    "MethylationClassifier",
    "ClassificationResult",
    "ReadClassification",
    # Training
    "BaumWelchTrainer",
    "train_from_tsvs",
    # Evaluation
    "Evaluator",
    "EvaluationMetrics",
    "load_ground_truth_labels",
]
