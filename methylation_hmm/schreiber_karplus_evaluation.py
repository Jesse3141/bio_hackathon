"""
Schreiber-Karplus Style Evaluation for Methylation HMM.

Adapts the original paper's evaluation methodology to our two-HMM classifier.
See TESTING_METHODOLOGY.md for the original metrics.

Key adaptations:
- Filter Score → abs(log ratio) as confidence proxy
- Soft Call → prediction correctness (0 or 1)
- MCSC → Mean Cumulative Accuracy (MCA)

The original paper used a label fork (T/X/CAT) for independent validation.
Since we lack this, we use log-likelihood ratio confidence as a proxy for
"on-pathway" vs "off-pathway" events.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass
class SchreiberKarplusMetrics:
    """Metrics following the Schreiber-Karplus methodology."""

    # Core metrics
    n_samples: int = 0
    overall_accuracy: float = 0.0

    # Confidence-based metrics (analogs to Filter Score / MCSC)
    mca_curve: np.ndarray = field(default_factory=lambda: np.array([]))  # Mean Cumulative Accuracy
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    # Key thresholds
    accuracy_at_25pct: float = 0.0  # Accuracy using top 25% (paper's key metric)
    accuracy_at_50pct: float = 0.0
    confidence_for_95pct_acc: float = 0.0  # Confidence needed for 95% accuracy
    on_pathway_fraction: float = 0.0  # Fraction with high confidence

    # ROC metrics
    roc_auc: float = 0.0
    average_precision: float = 0.0

    # Per-class
    control_accuracy: float = 0.0
    modified_accuracy: float = 0.0

    # Cross-validation (if applicable)
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0


@dataclass
class ReadResult:
    """Per-read classification result."""
    read_id: str
    true_label: int  # 0=control, 1=modified
    predicted_label: int
    confidence: float  # abs(log ratio) - analog to Filter Score
    correct: int  # 0 or 1 - analog to Soft Call


class SchreiberKarplusEvaluator:
    """
    Evaluate HMM classifiers using Schreiber-Karplus methodology.

    Maps our metrics to the original paper's framework:

    Original          | Our Analog
    ------------------|------------------
    Filter Score      | abs(log P(mod) - log P(ctrl))
    Soft Call         | 1 if correct, 0 if wrong
    MCSC              | Mean Cumulative Accuracy (MCA)
    Threshold 0.1     | Confidence percentile cutoff
    """

    def __init__(self, classifier):
        """
        Initialize evaluator.

        Args:
            classifier: Fitted SimplifiedMethylationClassifier or ForkHMMClassifier
        """
        self.classifier = classifier
        self.results: List[ReadResult] = []

    def score_reads(self, df: pd.DataFrame) -> List[ReadResult]:
        """
        Score all reads and store results.

        Args:
            df: DataFrame with position columns and 'sample' ground truth

        Returns:
            List of ReadResult objects sorted by confidence (descending)
        """
        POSITIONS = ['38', '50', '62', '74', '86', '98', '110', '122']

        df_valid = df.dropna(subset=POSITIONS)
        X = df_valid[POSITIONS].values.astype(np.float32)
        y_true = (df_valid['sample'].values == 'modified').astype(int)

        # Get predictions and confidences
        logp_control, logp_modified = self.classifier.predict_proba(X)
        y_pred = (logp_modified > logp_control).astype(int)
        confidences = np.abs(logp_modified - logp_control)

        read_ids = df_valid['read_id'].values if 'read_id' in df_valid.columns else \
                   [f'read_{i}' for i in range(len(X))]

        self.results = []
        for i in range(len(X)):
            self.results.append(ReadResult(
                read_id=str(read_ids[i]),
                true_label=int(y_true[i]),
                predicted_label=int(y_pred[i]),
                confidence=float(confidences[i]),
                correct=int(y_pred[i] == y_true[i])
            ))

        # Sort by confidence (descending) - same as paper sorts by Filter Score
        self.results.sort(key=lambda r: r.confidence, reverse=True)

        return self.results

    def compute_mca_curve(self) -> np.ndarray:
        """
        Compute Mean Cumulative Accuracy curve.

        This is our analog to MCSC (Mean Cumulative Soft Call).

        MCA[i] = mean accuracy using events ranked i and better (by confidence)

        Returns:
            Array of MCA values, one per read
        """
        if not self.results:
            raise ValueError("No results. Call score_reads() first.")

        n = len(self.results)
        correct = np.array([r.correct for r in self.results])

        # MCA[i] = mean(correct[i:])
        mca = np.array([correct[i:].mean() for i in range(n)])

        return mca

    def compute_metrics(self, df: pd.DataFrame) -> SchreiberKarplusMetrics:
        """
        Compute full Schreiber-Karplus style metrics.

        Args:
            df: DataFrame with position columns and 'sample' ground truth

        Returns:
            SchreiberKarplusMetrics object
        """
        # Score all reads
        self.score_reads(df)
        n = len(self.results)

        if n == 0:
            return SchreiberKarplusMetrics()

        # Basic accuracy
        correct = np.array([r.correct for r in self.results])
        y_true = np.array([r.true_label for r in self.results])
        y_pred = np.array([r.predicted_label for r in self.results])
        confidences = np.array([r.confidence for r in self.results])

        # MCA curve
        mca = self.compute_mca_curve()

        # Key thresholds (paper reports accuracy at top 26%)
        idx_25 = max(1, int(n * 0.25))
        idx_50 = max(1, int(n * 0.50))

        accuracy_at_25pct = correct[:idx_25].mean()
        accuracy_at_50pct = correct[:idx_50].mean()

        # Find confidence needed for 95% accuracy
        confidence_for_95 = 0.0
        for i in range(n):
            if correct[:i+1].mean() < 0.95:
                confidence_for_95 = confidences[i]
                break

        # On-pathway fraction: what fraction has "high" confidence?
        # Paper uses filter score > 0.1 which kept ~26% of events
        # We use the confidence at the 26th percentile as our threshold
        threshold_idx = int(n * 0.26)
        on_pathway_threshold = confidences[threshold_idx] if threshold_idx < n else 0
        on_pathway_fraction = (confidences > on_pathway_threshold).mean()

        # ROC metrics
        roc_auc = 0.0
        avg_precision = 0.0
        try:
            roc_auc = roc_auc_score(y_true, confidences * (2 * y_pred - 1))  # signed confidence
            avg_precision = average_precision_score(y_true, confidences * (2 * y_pred - 1))
        except:
            pass

        # Per-class accuracy
        control_mask = y_true == 0
        modified_mask = y_true == 1
        control_acc = (y_pred[control_mask] == 0).mean() if control_mask.sum() > 0 else 0.0
        modified_acc = (y_pred[modified_mask] == 1).mean() if modified_mask.sum() > 0 else 0.0

        return SchreiberKarplusMetrics(
            n_samples=n,
            overall_accuracy=correct.mean(),
            mca_curve=mca,
            confidence_scores=confidences,
            accuracy_at_25pct=accuracy_at_25pct,
            accuracy_at_50pct=accuracy_at_50pct,
            confidence_for_95pct_acc=confidence_for_95,
            on_pathway_fraction=on_pathway_fraction,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            control_accuracy=control_acc,
            modified_accuracy=modified_acc
        )

    def n_fold_cross_validation(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        n_iterations: int = 10,
        seed: int = 42
    ) -> SchreiberKarplusMetrics:
        """
        Run n-fold cross-validation repeated multiple times.

        Follows the paper's methodology: 5-fold CV repeated 10 times with shuffling.

        Args:
            df: Full dataset
            n_folds: Number of folds (default 5)
            n_iterations: Number of times to repeat CV (default 10)
            seed: Random seed

        Returns:
            SchreiberKarplusMetrics with CV statistics
        """
        random.seed(seed)
        np.random.seed(seed)

        POSITIONS = ['38', '50', '62', '74', '86', '98', '110', '122']
        df_valid = df.dropna(subset=POSITIONS).copy()

        all_accuracies = []
        all_aucs = []
        all_mca_curves = []

        for iteration in range(n_iterations):
            # Shuffle data
            df_shuffled = df_valid.sample(frac=1, random_state=seed + iteration)

            fold_accuracies = []
            fold_aucs = []

            for fold in range(n_folds):
                # Split into train/test
                fold_size = len(df_shuffled) // n_folds
                test_start = fold * fold_size
                test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(df_shuffled)

                test_df = df_shuffled.iloc[test_start:test_end]
                train_df = pd.concat([
                    df_shuffled.iloc[:test_start],
                    df_shuffled.iloc[test_end:]
                ])

                # Train fresh model (import here to avoid circular deps)
                from .simplified_pipeline import SimplifiedMethylationClassifier

                # Split training by class
                train_control = train_df[train_df['sample'] == 'control']
                train_modified = train_df[train_df['sample'] == 'modified']

                if len(train_control) == 0 or len(train_modified) == 0:
                    continue

                # Create and train classifier
                classifier = SimplifiedMethylationClassifier()
                classifier._compute_emission_params(train_control, train_modified)
                classifier._build_models_from_params()

                X_control = torch.tensor(
                    train_control[POSITIONS].values.astype(np.float32)
                ).unsqueeze(-1)
                X_modified = torch.tensor(
                    train_modified[POSITIONS].values.astype(np.float32)
                ).unsqueeze(-1)

                classifier.control_model.fit(X_control)
                classifier.modified_model.fit(X_modified)
                classifier.is_fitted = True

                # Evaluate on test fold
                evaluator = SchreiberKarplusEvaluator(classifier)
                metrics = evaluator.compute_metrics(test_df)

                fold_accuracies.append(metrics.overall_accuracy)
                fold_aucs.append(metrics.roc_auc)
                all_mca_curves.append(metrics.mca_curve)

            all_accuracies.extend(fold_accuracies)
            all_aucs.extend(fold_aucs)

        # Aggregate results
        result = SchreiberKarplusMetrics(
            n_samples=len(df_valid),
            overall_accuracy=np.mean(all_accuracies),
            cv_accuracy_mean=np.mean(all_accuracies),
            cv_accuracy_std=np.std(all_accuracies),
            cv_auc_mean=np.mean(all_aucs),
            cv_auc_std=np.std(all_aucs)
        )

        return result


def print_schreiber_karplus_report(metrics: SchreiberKarplusMetrics) -> None:
    """Print a formatted report in Schreiber-Karplus style."""

    print("\n" + "=" * 60)
    print("SCHREIBER-KARPLUS STYLE EVALUATION REPORT")
    print("=" * 60)

    print(f"\nDataset: {metrics.n_samples} reads")

    print("\n--- Overall Performance ---")
    print(f"  Accuracy (all reads):     {metrics.overall_accuracy:.1%}")
    print(f"  Control class accuracy:   {metrics.control_accuracy:.1%}")
    print(f"  Modified class accuracy:  {metrics.modified_accuracy:.1%}")

    print("\n--- Confidence-Stratified Accuracy (Key Paper Metrics) ---")
    print(f"  Top 25% (high confidence): {metrics.accuracy_at_25pct:.1%}")
    print(f"  Top 50% (medium):          {metrics.accuracy_at_50pct:.1%}")
    print(f"  All reads (100%):          {metrics.overall_accuracy:.1%}")

    if metrics.confidence_for_95pct_acc > 0:
        print(f"\n  Confidence threshold for 95% accuracy: {metrics.confidence_for_95pct_acc:.2f}")

    print(f"\n--- On-Pathway Analysis ---")
    print(f"  'On-pathway' fraction (top 26%): {metrics.on_pathway_fraction:.1%}")

    print(f"\n--- ROC Metrics ---")
    print(f"  ROC AUC:            {metrics.roc_auc:.3f}")
    print(f"  Average Precision:  {metrics.average_precision:.3f}")

    if metrics.cv_accuracy_mean > 0:
        print(f"\n--- Cross-Validation Results ---")
        print(f"  CV Accuracy: {metrics.cv_accuracy_mean:.1%} +/- {metrics.cv_accuracy_std:.1%}")
        print(f"  CV AUC:      {metrics.cv_auc_mean:.3f} +/- {metrics.cv_auc_std:.3f}")

    print("\n" + "=" * 60)

    # Comparison to paper
    print("\n--- Comparison to Schreiber-Karplus Paper ---")
    print("| Metric                    | Paper   | This Model |")
    print("|---------------------------|---------|------------|")
    print(f"| Accuracy @ top 26%        | 97.7%   | {metrics.accuracy_at_25pct:.1%}      |")
    print(f"| Overall accuracy          | ~75-80% | {metrics.overall_accuracy:.1%}      |")
    print(f"| On-pathway fraction       | ~26%    | {metrics.on_pathway_fraction:.1%}       |")
    print("=" * 60 + "\n")


def generate_mca_plot_data(metrics: SchreiberKarplusMetrics) -> pd.DataFrame:
    """
    Generate data for MCA curve plot (analog to paper's Figure 4).

    Returns:
        DataFrame with columns: event_rank, coverage, mca
    """
    n = len(metrics.mca_curve)
    return pd.DataFrame({
        'event_rank': np.arange(n),
        'coverage': np.linspace(0, 1, n),
        'mca': metrics.mca_curve,
        'confidence': metrics.confidence_scores
    })


def compare_to_paper_baseline() -> Dict[str, float]:
    """
    Return the paper's baseline metrics for comparison.
    """
    return {
        'accuracy_at_26pct': 0.977,  # 97.7%
        'overall_accuracy': 0.77,    # ~75-80%
        'on_pathway_fraction': 0.26,  # 26%
        'filter_threshold': 0.10,     # Filter score threshold
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/jesse/repos/bio_hackathon')

    from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier

    # Load data
    df = pd.read_csv('output/hmm_training_sequences.csv')

    # Load pre-trained classifier
    classifier = SimplifiedMethylationClassifier.from_json(
        'output/hmm_emission_params_pomegranate.json'
    )

    # Evaluate
    evaluator = SchreiberKarplusEvaluator(classifier)
    metrics = evaluator.compute_metrics(df)

    print_schreiber_karplus_report(metrics)

    # Save MCA curve data
    mca_data = generate_mca_plot_data(metrics)
    mca_data.to_csv('output/mca_curve_data.csv', index=False)
    print(f"MCA curve data saved to output/mca_curve_data.csv")
