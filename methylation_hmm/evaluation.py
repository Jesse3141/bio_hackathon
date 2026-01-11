"""
Evaluation metrics for methylation classification.

Computes accuracy, AUC, confusion matrices, and confidence-stratified
accuracy curves.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    n_samples: int
    n_correct: int


class Evaluator:
    """Evaluate classification performance."""

    def __init__(self, cytosine_positions: Tuple[int, ...] = None):
        """
        Initialize evaluator.

        Args:
            cytosine_positions: Positions to evaluate (default: all 8)
        """
        if cytosine_positions is None:
            cytosine_positions = (38, 50, 62, 74, 86, 98, 110, 122)
        self.cytosine_positions = cytosine_positions

    def compute_metrics(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> EvaluationMetrics:
        """
        Compute classification metrics.

        Args:
            predictions: DataFrame with 'read_id', 'position', 'call'
            ground_truth: DataFrame with 'read_id', 'position', 'label'

        Returns:
            EvaluationMetrics object
        """
        # Merge predictions with ground truth
        merged = predictions.merge(
            ground_truth,
            on=['read_id', 'position'],
            how='inner'
        )

        if len(merged) == 0:
            return EvaluationMetrics(
                accuracy=0, precision=0, recall=0, f1_score=0,
                auc_roc=None, n_samples=0, n_correct=0
            )

        # Convert to binary: 5mC = 1, C = 0
        y_true = (merged['label'] == '5mC').astype(int).values
        y_pred = (merged['call'] == '5mC').astype(int).values

        # Metrics
        n_samples = len(y_true)
        n_correct = (y_true == y_pred).sum()
        accuracy = n_correct / n_samples

        # Precision, recall, F1 for 5mC class
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # AUC-ROC if probabilities available
        auc_roc = None
        if 'p_methylated' in merged.columns:
            try:
                from sklearn.metrics import roc_auc_score
                auc_roc = roc_auc_score(y_true, merged['p_methylated'].values)
            except:
                pass

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            n_samples=n_samples,
            n_correct=n_correct
        )

    def accuracy_vs_confidence_curve(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        n_thresholds: int = 20
    ) -> pd.DataFrame:
        """
        Compute accuracy at different confidence thresholds.

        Replicates the paper's "top X% accuracy" analysis.

        Args:
            predictions: DataFrame with predictions and filter_score
            ground_truth: DataFrame with labels
            n_thresholds: Number of threshold points

        Returns:
            DataFrame with columns: threshold, coverage, accuracy
        """
        merged = predictions.merge(
            ground_truth,
            on=['read_id', 'position'],
            how='inner'
        )

        if len(merged) == 0:
            return pd.DataFrame(columns=['threshold', 'coverage', 'accuracy'])

        # Sort by filter_score descending (highest confidence first)
        merged = merged.sort_values('filter_score', ascending=False)

        results = []
        n_total = len(merged)

        thresholds = np.linspace(0, 1, n_thresholds)

        for thresh in thresholds:
            # Keep top (1 - thresh) fraction
            n_keep = int(n_total * (1 - thresh))
            if n_keep == 0:
                continue

            subset = merged.head(n_keep)

            y_true = (subset['label'] == '5mC').astype(int)
            y_pred = (subset['call'] == '5mC').astype(int)

            accuracy = (y_true == y_pred).mean()
            coverage = n_keep / n_total

            results.append({
                'threshold': thresh,
                'coverage': coverage,
                'accuracy': accuracy
            })

        return pd.DataFrame(results)

    def per_position_accuracy(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> Dict[int, float]:
        """
        Compute accuracy for each cytosine position.

        Args:
            predictions: DataFrame with predictions
            ground_truth: DataFrame with labels

        Returns:
            {position: accuracy}
        """
        merged = predictions.merge(
            ground_truth,
            on=['read_id', 'position'],
            how='inner'
        )

        results = {}

        for pos in self.cytosine_positions:
            subset = merged[merged['position'] == pos]
            if len(subset) == 0:
                results[pos] = 0.0
                continue

            y_true = (subset['label'] == '5mC').astype(int)
            y_pred = (subset['call'] == '5mC').astype(int)
            results[pos] = (y_true == y_pred).mean()

        return results

    def confusion_matrix(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate confusion matrix.

        Args:
            predictions: DataFrame with predictions
            ground_truth: DataFrame with labels

        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        merged = predictions.merge(
            ground_truth,
            on=['read_id', 'position'],
            how='inner'
        )

        y_true = (merged['label'] == '5mC').astype(int).values
        y_pred = (merged['call'] == '5mC').astype(int).values

        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()

        return np.array([[tn, fp], [fn, tp]])

    def print_report(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> None:
        """
        Print a formatted evaluation report.

        Args:
            predictions: DataFrame with predictions
            ground_truth: DataFrame with labels
        """
        metrics = self.compute_metrics(predictions, ground_truth)
        cm = self.confusion_matrix(predictions, ground_truth)
        per_pos = self.per_position_accuracy(predictions, ground_truth)

        print("\n" + "="*50)
        print("METHYLATION CLASSIFICATION REPORT")
        print("="*50)

        print(f"\nOverall Metrics (n={metrics.n_samples}):")
        print(f"  Accuracy:  {metrics.accuracy:.3f}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall:    {metrics.recall:.3f}")
        print(f"  F1 Score:  {metrics.f1_score:.3f}")
        if metrics.auc_roc:
            print(f"  AUC-ROC:   {metrics.auc_roc:.3f}")

        print("\nConfusion Matrix:")
        print("              Predicted")
        print("              C     5mC")
        print(f"  Actual C   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"  Actual 5mC {cm[1,0]:5d}  {cm[1,1]:5d}")

        print("\nPer-Position Accuracy:")
        for pos, acc in per_pos.items():
            print(f"  Position {pos:3d}: {acc:.3f}")

        print("="*50 + "\n")


def load_ground_truth_labels(
    labels_path: str,
    read_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load ground truth labels from BED-like file.

    Expected format: chrom, start, end, mod_type, score, strand
    Where mod_type: '-' = C, 'm' = 5mC, 'h' = 5hmC

    Args:
        labels_path: Path to labels file
        read_ids: Optional list of read IDs to include

    Returns:
        DataFrame with columns: read_id, position, label
    """
    df = pd.read_csv(
        labels_path,
        sep='\t',
        names=['chrom', 'start', 'end', 'mod_type', 'score', 'strand'],
        header=None
    )

    # Map modification codes
    mod_map = {'-': 'C', 'm': '5mC', 'h': '5hmC', '0': 'C', '1': '5mC', '2': '5hmC'}

    records = []
    for _, row in df.iterrows():
        label = mod_map.get(str(row['mod_type']), 'C')

        # If read_ids provided, filter
        if read_ids is not None and row['chrom'] not in read_ids:
            continue

        records.append({
            'read_id': row['chrom'],
            'position': int(row['start']),
            'label': label
        })

    return pd.DataFrame(records)
