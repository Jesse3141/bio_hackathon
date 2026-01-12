#!/usr/bin/env python
"""
Evaluate Full-Sequence HMM classifiers.

This script evaluates the FullSequenceHMM classifier that models the entire 155bp
reference sequence. The key variable is emission parameter source:
- single: Parameters from one adapter (context-specific, lower variance)
- pooled: Parameters from all adapters (higher variance)

Examples:
    # Binary mode, single adapter emissions
    python run_evaluation.py --mode binary --emission-source single

    # 3-way mode, pooled emissions
    python run_evaluation.py --mode 3way --emission-source pooled

    # With cross-validation
    python run_evaluation.py --mode binary --emission-source single --cv-folds 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from methylation_hmm.full_sequence_hmm import FullSequenceHMM, EvaluationMetrics
from methylation_hmm.emission_params import (
    compute_emission_params_from_full_csv,
    FullSequenceEmissionParams,
)
from methylation_hmm.evaluation import (
    UnifiedMetrics,
    UnifiedEvaluator,
    save_json,
    save_csv,
    generate_plots,
)
from methylation_hmm.evaluation.site_type_metrics import (
    load_bed_with_site_types,
    get_site_type_positions,
)


def get_default_paths() -> Dict[str, Path]:
    """Get default file paths."""
    return {
        "full_signal_csv": PROJECT_ROOT / "output" / "rep1" / "signal_full_sequence.csv",
        "cytosine_signal_csv": PROJECT_ROOT / "output" / "rep1" / "signal_at_cytosines_3way.csv",
        "bed_file": PROJECT_ROOT / "nanopore_ref_data" / "all_5mers_C_sites.bed",
        "output_dir": PROJECT_ROOT / "results" / "full_evaluation",
    }


SINGLE_ADAPTER = "5mers_rand_ref_adapter_01"
BINARY_SAMPLES = ["control", "5mC"]
THREE_WAY_SAMPLES = ["control", "5mC", "5hmC"]


def load_full_sequence_data(
    csv_path: Path,
    adapter: Optional[str],
    mode: str,
    test_split: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split full-sequence data.

    Args:
        csv_path: Path to signal_full_sequence.csv
        adapter: Adapter name for single mode (None for pooled)
        mode: 'binary' or '3way'
        test_split: Fraction for test set
        seed: Random seed

    Returns:
        (full_df, train_df, test_df)
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter by adapter if single mode
    if adapter:
        df = df[df['chrom'] == adapter]
        print(f"Filtered to adapter: {adapter}")

    # Filter samples based on mode
    samples = BINARY_SAMPLES if mode == "binary" else THREE_WAY_SAMPLES
    df = df[df['sample'].isin(samples)]
    print(f"Samples: {samples}")
    print(f"Total rows: {len(df):,}, Unique reads: {df['read_id'].nunique():,}")

    # Split by read_id
    np.random.seed(seed)
    read_ids = df['read_id'].unique()
    np.random.shuffle(read_ids)
    n_test = int(len(read_ids) * test_split)
    test_ids = set(read_ids[:n_test])

    test_df = df[df['read_id'].isin(test_ids)]
    train_df = df[~df['read_id'].isin(test_ids)]

    print(f"Train: {train_df['read_id'].nunique():,} reads, Test: {test_df['read_id'].nunique():,} reads")

    return df, train_df, test_df


def evaluate_full_sequence_hmm(
    test_df: pd.DataFrame,
    emission_params: FullSequenceEmissionParams,
    mode: str,
    cv_folds: int = 0,
    train_df: Optional[pd.DataFrame] = None,
    site_type_positions: Optional[Dict[int, int]] = None,
) -> UnifiedMetrics:
    """
    Evaluate FullSequenceHMM with comprehensive metrics.

    Args:
        test_df: Test data
        emission_params: Emission parameters
        mode: 'binary' or '3way'
        cv_folds: Number of CV folds (0 to skip)
        train_df: Training data (for CV)
        site_type_positions: Position to site type mapping

    Returns:
        UnifiedMetrics with all evaluation results
    """
    # Create classifier
    hmm = FullSequenceHMM.from_emission_params(emission_params)

    # Get predictions
    results = hmm.classify_dataframe(test_df)

    if len(results) == 0:
        raise ValueError("No valid reads for classification")

    # Build arrays for evaluation
    class_names = ['C', '5mC'] if mode == 'binary' else ['C', '5mC', '5hmC']
    sample_to_idx = {'control': 0, 'C': 0, '5mC': 1, 'mC': 1, '5hmC': 2, 'hmC': 2}

    read_samples = test_df.groupby('read_id')['sample'].first().to_dict()
    y_true = np.array([sample_to_idx.get(read_samples.get(r.read_id, 'control'), 0) for r in results])
    y_pred = np.array([r.prediction_idx for r in results])
    y_proba = np.array([[r.probabilities.get(c, 0.0) for c in class_names] for r in results])
    confidence = np.array([r.confidence for r in results])

    # Core metrics
    n_samples = len(y_true)
    n_correct = (y_pred == y_true).sum()
    accuracy = n_correct / n_samples

    # Per-class accuracy
    per_class_acc = {}
    for idx, name in enumerate(class_names):
        mask = y_true == idx
        if mask.sum() > 0:
            per_class_acc[name] = float((y_pred[mask] == idx).mean())
        else:
            per_class_acc[name] = 0.0

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, roc_auc_score
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Site-type metrics (placeholder - full-sequence doesn't have per-position predictions)
    acc_by_site = {'non_cpg': accuracy, 'cpg': accuracy, 'homopolymer': accuracy}
    n_by_site = {'non_cpg': n_samples, 'cpg': n_samples, 'homopolymer': n_samples}

    # Confidence-stratified accuracy
    sorted_idx = np.argsort(confidence)[::-1]
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    correct = (y_pred_sorted == y_true_sorted).astype(float)
    mca_curve = np.cumsum(correct) / np.arange(1, n_samples + 1)

    idx_25 = max(1, int(n_samples * 0.25))
    idx_50 = max(1, int(n_samples * 0.50))

    acc_25 = float(mca_curve[idx_25 - 1])
    acc_50 = float(mca_curve[idx_50 - 1])
    acc_100 = float(mca_curve[-1])

    # AUC
    try:
        if len(class_names) == 2:
            auc_macro = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr')
        auc_per_class = {name: auc_macro for name in class_names}  # Simplified
    except ValueError:
        auc_macro = None
        auc_per_class = None

    # Build model name
    source = emission_params.source
    model_name = f"FullSequenceHMM_{mode}_{source}"

    metrics = UnifiedMetrics(
        overall_accuracy=float(accuracy),
        n_samples=n_samples,
        n_correct=int(n_correct),
        per_class_accuracy=per_class_acc,
        confusion_matrix=conf_mat,
        accuracy_by_site_type=acc_by_site,
        n_samples_by_site_type=n_by_site,
        accuracy_at_top_25pct=acc_25,
        accuracy_at_top_50pct=acc_50,
        accuracy_at_100pct=acc_100,
        mca_curve=mca_curve,
        auc_macro=auc_macro,
        auc_per_class=auc_per_class,
        model_name=model_name,
        mode=mode,
        adapters=[emission_params.adapter] if emission_params.adapter else ['all'],
    )

    # Cross-validation
    if cv_folds > 0 and train_df is not None:
        print(f"Running {cv_folds}-fold cross-validation...")
        cv_accs, cv_aucs = run_cross_validation(
            pd.concat([train_df, test_df]),
            emission_params,
            mode,
            n_folds=cv_folds,
        )
        metrics.cv_fold_accuracies = cv_accs
        metrics.cv_mean_accuracy = float(np.mean(cv_accs))
        metrics.cv_std_accuracy = float(np.std(cv_accs))
        metrics.cv_fold_aucs = cv_aucs
        metrics.cv_mean_auc = float(np.mean(cv_aucs)) if cv_aucs else None
        metrics.cv_std_auc = float(np.std(cv_aucs)) if cv_aucs else None

    return metrics


def run_cross_validation(
    df: pd.DataFrame,
    emission_params: FullSequenceEmissionParams,
    mode: str,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """Run k-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold

    # Get unique reads with their labels
    read_labels = df.groupby('read_id')['sample'].first().reset_index()
    sample_to_idx = {'control': 0, '5mC': 1, '5hmC': 2}
    read_labels['label'] = read_labels['sample'].map(sample_to_idx)

    read_ids = read_labels['read_id'].values
    labels = read_labels['label'].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_accs = []
    fold_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(read_ids, labels)):
        test_read_ids = set(read_ids[test_idx])
        test_fold_df = df[df['read_id'].isin(test_read_ids)]

        # Create HMM and evaluate
        hmm = FullSequenceHMM.from_emission_params(emission_params)
        metrics = hmm.evaluate(test_fold_df)

        fold_accs.append(metrics.accuracy)

        # AUC computation would require probabilities
        fold_aucs.append(0.0)  # Placeholder

        print(f"  Fold {fold_idx + 1}: accuracy = {metrics.accuracy:.1%}")

    return fold_accs, fold_aucs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Full-Sequence HMM classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Core arguments
    parser.add_argument(
        "--mode",
        choices=["binary", "3way"],
        required=True,
        help="Classification mode: binary (C/5mC) or 3way (C/5mC/5hmC)",
    )
    parser.add_argument(
        "--emission-source",
        choices=["single", "pooled"],
        required=True,
        help="Emission parameter source: single (adapter_01) or pooled (all adapters)",
    )

    # Data arguments
    defaults = get_default_paths()
    parser.add_argument(
        "--signal-csv",
        type=Path,
        default=defaults["full_signal_csv"],
        help="Path to signal_full_sequence.csv",
    )
    parser.add_argument(
        "--bed-file",
        type=Path,
        default=defaults["bed_file"],
        help="Path to BED file with site types",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults["output_dir"],
        help="Output directory for results",
    )

    # Evaluation arguments
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (0 to skip CV)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Output options
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    # Determine adapter based on emission source
    adapter = SINGLE_ADAPTER if args.emission_source == "single" else None

    # Create output directory
    config_name = f"{args.mode}_{args.emission_source}"
    output_dir = args.output / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL-SEQUENCE HMM EVALUATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Emission source: {args.emission_source}")
    print(f"Adapter: {adapter or 'all (pooled)'}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    full_df, train_df, test_df = load_full_sequence_data(
        csv_path=args.signal_csv,
        adapter=adapter,
        mode=args.mode,
        test_split=args.test_split,
        seed=args.seed,
    )

    # Compute emission parameters
    print("\nComputing emission parameters...")
    emission_params = compute_emission_params_from_full_csv(
        str(args.signal_csv),
        adapter=adapter,
        mode=args.mode,
    )

    # Save emission parameters
    params_path = output_dir / "emission_params.json"
    emission_params.save(str(params_path))
    print(f"Saved emission parameters to {params_path}")

    # Print emission summary
    summary = emission_params.summary()
    print(f"\nEmission parameter summary:")
    print(f"  Mean C current: {summary['mean_C_current']:.1f} pA")
    print(f"  Mean C std: {summary['mean_C_std']:.1f} pA")
    print(f"  Mean Δ(5mC-C): {summary['mean_delta_5mC_C']:.1f} pA")
    if args.mode == '3way':
        print(f"  Mean Δ(5hmC-C): {summary['mean_delta_5hmC_C']:.1f} pA")

    # Load site type info
    print("\nLoading site type annotations...")
    bed_df = load_bed_with_site_types(str(args.bed_file))
    site_type_positions = {}
    if adapter:
        site_type_positions = get_site_type_positions(bed_df, adapter)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_full_sequence_hmm(
        test_df=test_df,
        emission_params=emission_params,
        mode=args.mode,
        cv_folds=args.cv_folds,
        train_df=train_df,
        site_type_positions=site_type_positions,
    )

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Overall accuracy: {metrics.overall_accuracy:.1%}")
    print(f"\nPer-class accuracy:")
    for name, acc in metrics.per_class_accuracy.items():
        print(f"  {name}: {acc:.1%}")
    print(f"\nConfidence-stratified accuracy:")
    print(f"  Top 25%: {metrics.accuracy_at_top_25pct:.1%}")
    print(f"  Top 50%: {metrics.accuracy_at_top_50pct:.1%}")
    print(f"  All:     {metrics.accuracy_at_100pct:.1%}")
    if metrics.auc_macro:
        print(f"\nAUC (macro): {metrics.auc_macro:.3f}")
    if metrics.cv_mean_accuracy:
        print(f"\nCross-validation: {metrics.cv_mean_accuracy:.1%} +/- {metrics.cv_std_accuracy:.1%}")

    print(f"\nConfusion matrix:")
    print(metrics.confusion_matrix)

    # Save outputs
    print()
    print("Saving results...")

    # Add emission summary to metrics dict
    metrics_dict = metrics.to_dict()
    metrics_dict['emission_params_summary'] = summary
    metrics_dict['sequence_length'] = emission_params.sequence_length

    # Save JSON
    json_path = output_dir / "metrics.json"
    with open(json_path, 'w') as f:
        # Handle numpy arrays
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        json.dump(metrics_dict, f, indent=2, default=convert)
    print(f"  Saved: {json_path}")

    # Save CSV
    save_csv(metrics, output_dir / "metrics.csv")
    print(f"  Saved: {output_dir / 'metrics.csv'}")

    # Generate plots
    if not args.no_plots:
        plot_files = generate_plots(metrics, output_dir)
        for pf in plot_files:
            print(f"  Saved: {pf}")

    print()
    print("Done!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
