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
from .evaluation_legacy import Evaluator, EvaluationMetrics, load_ground_truth_labels
from .simplified_pipeline import (
    SimplifiedMethylationClassifier,
    ClassificationResult as SimplifiedClassificationResult,
    EvaluationMetrics as SimplifiedEvaluationMetrics,
    run_full_pipeline,
)
from .schreiber_karplus_evaluation import (
    SchreiberKarplusEvaluator,
    SchreiberKarplusMetrics,
    print_schreiber_karplus_report,
    generate_mca_plot_data,
    compare_to_paper_baseline,
)
from .generic_hmm import (
    GenericHMM,
    ClassificationResult as GenericClassificationResult,
    EvaluationMetrics as GenericEvaluationMetrics,
    run_evaluation as run_generic_evaluation,
)
from .full_sequence_hmm import (
    FullSequenceHMM,
    ClassificationResult as FullSequenceClassificationResult,
    EvaluationMetrics as FullSequenceEvaluationMetrics,
    run_evaluation as run_full_sequence_evaluation,
)
from .emission_params import (
    FullSequenceEmissionParams,
    EmissionParams,
    PositionEmissions,
    compute_emission_params_from_full_csv,
    compute_emission_params_from_cytosine_csv,
    compare_emission_sources,
)

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
    # Simplified Pipeline (recommended for POD5-derived data)
    "SimplifiedMethylationClassifier",
    "SimplifiedClassificationResult",
    "SimplifiedEvaluationMetrics",
    "run_full_pipeline",
    # Schreiber-Karplus Style Evaluation
    "SchreiberKarplusEvaluator",
    "SchreiberKarplusMetrics",
    "print_schreiber_karplus_report",
    "generate_mca_plot_data",
    "compare_to_paper_baseline",
    # Generic 3-way HMM (C / 5mC / 5hmC)
    "GenericHMM",
    "GenericClassificationResult",
    "GenericEvaluationMetrics",
    "run_generic_evaluation",
    # Full-Sequence HMM (entire 155bp sequence)
    "FullSequenceHMM",
    "FullSequenceClassificationResult",
    "FullSequenceEvaluationMetrics",
    "run_full_sequence_evaluation",
    # Emission Parameters
    "FullSequenceEmissionParams",
    "EmissionParams",
    "PositionEmissions",
    "compute_emission_params_from_full_csv",
    "compute_emission_params_from_cytosine_csv",
    "compare_emission_sources",
]
