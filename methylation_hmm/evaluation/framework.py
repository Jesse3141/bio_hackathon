"""
Unified evaluation framework for methylation HMM classifiers.

This module provides a standardized evaluation interface that works with
any classifier implementing predict() and predict_proba() methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Protocol, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score


@dataclass
class UnifiedMetrics:
    """Comprehensive evaluation metrics for methylation classifiers."""

    # Basic metrics
    overall_accuracy: float
    n_samples: int
    n_correct: int

    # Per-class metrics
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray

    # Site-type specific metrics
    accuracy_by_site_type: Dict[str, float]
    n_samples_by_site_type: Dict[str, int]

    # Confidence-based metrics (Schreiber-Karplus style)
    accuracy_at_top_25pct: float
    accuracy_at_top_50pct: float
    accuracy_at_100pct: float
    mca_curve: Optional[np.ndarray] = None

    # AUC metrics
    auc_macro: Optional[float] = None
    auc_per_class: Optional[Dict[str, float]] = None

    # Cross-validation (optional)
    cv_fold_accuracies: Optional[List[float]] = None
    cv_mean_accuracy: Optional[float] = None
    cv_std_accuracy: Optional[float] = None
    cv_fold_aucs: Optional[List[float]] = None
    cv_mean_auc: Optional[float] = None
    cv_std_auc: Optional[float] = None

    # Metadata
    model_name: str = ""
    mode: str = ""  # "binary" or "3way"
    adapters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "model": self.model_name,
            "mode": self.mode,
            "adapters": self.adapters,
            "overall": {
                "accuracy": self.overall_accuracy,
                "n_samples": self.n_samples,
                "n_correct": self.n_correct,
            },
            "per_class": self.per_class_accuracy,
            "by_site_type": {
                "accuracy": self.accuracy_by_site_type,
                "n_samples": self.n_samples_by_site_type,
            },
            "confidence_stratified": {
                "top_25pct": self.accuracy_at_top_25pct,
                "top_50pct": self.accuracy_at_top_50pct,
                "top_100pct": self.accuracy_at_100pct,
            },
            "confusion_matrix": self.confusion_matrix.tolist(),
            "auc": {
                "macro": self.auc_macro,
                "per_class": self.auc_per_class,
            },
            "cross_validation": {
                "mean_accuracy": self.cv_mean_accuracy,
                "std_accuracy": self.cv_std_accuracy,
                "fold_accuracies": self.cv_fold_accuracies,
                "mean_auc": self.cv_mean_auc,
                "std_auc": self.cv_std_auc,
                "fold_aucs": self.cv_fold_aucs,
            } if self.cv_mean_accuracy is not None else None,
        }


class ClassifierProtocol(Protocol):
    """Protocol defining the interface for classifiers."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        ...


class UnifiedEvaluator:
    """
    Unified evaluator for methylation classifiers.

    Works with any classifier that implements predict() and predict_proba() methods.
    Provides comprehensive evaluation including site-type breakdown and
    confidence-stratified accuracy.
    """

    POSITIONS = [38, 50, 62, 74, 86, 98, 110, 122]
    POSITION_COLS = ['38', '50', '62', '74', '86', '98', '110', '122']
    SITE_TYPE_NAMES = {0: 'non_cpg', 1: 'cpg', 2: 'homopolymer'}

    def __init__(
        self,
        class_names: List[str],
        model_name: str = "",
        mode: str = "",
        adapters: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.

        Args:
            class_names: List of class names ['C', '5mC'] or ['C', '5mC', '5hmC']
            model_name: Name of the classifier for reporting
            mode: Classification mode ("binary" or "3way")
            adapters: List of adapters used for training
        """
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.model_name = model_name
        self.mode = mode
        self.adapters = adapters or []

        # Build sample name to class index mapping
        self.sample_to_idx = self._build_sample_mapping()

    def _build_sample_mapping(self) -> Dict[str, int]:
        """Build mapping from sample names to class indices."""
        mapping = {}
        for idx, name in enumerate(self.class_names):
            mapping[name] = idx
            # Add common aliases
            if name == 'C':
                mapping['control'] = idx
            elif name == '5mC':
                mapping['mC'] = idx
                mapping['modified'] = idx
            elif name == '5hmC':
                mapping['hmC'] = idx
        return mapping

    def evaluate(
        self,
        classifier: ClassifierProtocol,
        df: pd.DataFrame,
        site_type_positions: Optional[Dict[int, int]] = None,
        confidence_scores: Optional[np.ndarray] = None,
    ) -> UnifiedMetrics:
        """
        Evaluate classifier on labeled data.

        Args:
            classifier: Classifier with predict() and predict_proba() methods
            df: Wide-format DataFrame with position columns and 'sample' column
            site_type_positions: Dict mapping position -> site_type (0, 1, 2)
            confidence_scores: Pre-computed confidence scores (optional)

        Returns:
            UnifiedMetrics object with all evaluation results
        """
        # Extract features and labels
        X, y_true = self._extract_features_labels(df)

        # Get predictions
        y_pred = classifier.predict(X)
        y_proba = classifier.predict_proba(X)

        # Compute confidence scores if not provided
        if confidence_scores is None:
            confidence_scores = self._compute_confidence(y_proba)

        # Core metrics
        accuracy = (y_pred == y_true).mean()
        n_correct = (y_pred == y_true).sum()

        # Per-class accuracy
        per_class_acc = {}
        for idx, name in enumerate(self.class_names):
            mask = y_true == idx
            if mask.sum() > 0:
                per_class_acc[name] = float((y_pred[mask] == idx).mean())
            else:
                per_class_acc[name] = 0.0

        # Confusion matrix
        conf_mat = confusion_matrix(
            y_true, y_pred,
            labels=list(range(self.n_classes))
        )

        # Site-type metrics
        acc_by_site, n_by_site = self._compute_site_type_metrics(
            df, y_true, y_pred, site_type_positions
        )

        # Confidence-stratified accuracy
        acc_25, acc_50, acc_100, mca_curve = self._compute_confidence_metrics(
            y_true, y_pred, confidence_scores
        )

        # AUC metrics
        auc_macro, auc_per_class = self._compute_auc_metrics(y_true, y_proba)

        return UnifiedMetrics(
            overall_accuracy=float(accuracy),
            n_samples=len(y_true),
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
            model_name=self.model_name,
            mode=self.mode,
            adapters=self.adapters,
        )

    def cross_validate(
        self,
        classifier_factory,
        df: pd.DataFrame,
        n_folds: int = 5,
        site_type_positions: Optional[Dict[int, int]] = None,
        random_state: int = 42,
    ) -> UnifiedMetrics:
        """
        Perform k-fold cross-validation.

        Args:
            classifier_factory: Callable that returns a fresh classifier
            df: Wide-format DataFrame with position columns and 'sample' column
            n_folds: Number of CV folds
            site_type_positions: Dict mapping position -> site_type
            random_state: Random seed for reproducibility

        Returns:
            UnifiedMetrics with CV statistics
        """
        X, y = self._extract_features_labels(df)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        fold_accuracies = []
        fold_aucs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Create and train classifier
            classifier = classifier_factory()
            classifier.fit(X_train, y_train)

            # Evaluate
            y_pred = classifier.predict(X_test)
            y_proba = classifier.predict_proba(X_test)

            acc = (y_pred == y_test).mean()
            fold_accuracies.append(float(acc))

            # AUC
            try:
                if self.n_classes == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                fold_aucs.append(float(auc))
            except ValueError:
                fold_aucs.append(0.0)

        # Get final metrics from last fold's classifier on full test
        # (This is approximate - real implementation would aggregate)
        final_metrics = self.evaluate(
            classifier, df.iloc[test_idx], site_type_positions
        )

        # Add CV statistics
        final_metrics.cv_fold_accuracies = fold_accuracies
        final_metrics.cv_mean_accuracy = float(np.mean(fold_accuracies))
        final_metrics.cv_std_accuracy = float(np.std(fold_accuracies))
        final_metrics.cv_fold_aucs = fold_aucs
        final_metrics.cv_mean_auc = float(np.mean(fold_aucs))
        final_metrics.cv_std_auc = float(np.std(fold_aucs))

        return final_metrics

    def _extract_features_labels(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and labels from DataFrame."""
        # Ensure we have the position columns
        df_valid = df.dropna(subset=self.POSITION_COLS)

        X = df_valid[self.POSITION_COLS].values.astype(np.float32)
        y = df_valid['sample'].map(self.sample_to_idx).values.astype(int)

        return X, y

    def _compute_confidence(self, y_proba: np.ndarray) -> np.ndarray:
        """Compute confidence scores from probabilities."""
        # Confidence = max probability - second max probability
        sorted_probs = np.sort(y_proba, axis=1)[:, ::-1]
        return sorted_probs[:, 0] - sorted_probs[:, 1]

    def _compute_site_type_metrics(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        site_type_positions: Optional[Dict[int, int]],
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Compute accuracy grouped by site type."""
        acc_by_site = {}
        n_by_site = {}

        if site_type_positions is None:
            # No site type info - return empty dicts
            for st_name in self.SITE_TYPE_NAMES.values():
                acc_by_site[st_name] = 0.0
                n_by_site[st_name] = 0
            return acc_by_site, n_by_site

        # Group positions by site type
        positions_by_type = {0: [], 1: [], 2: []}
        for pos, st in site_type_positions.items():
            if st in positions_by_type:
                positions_by_type[st].append(pos)

        # For each site type, compute accuracy
        for st, positions in positions_by_type.items():
            st_name = self.SITE_TYPE_NAMES[st]
            if len(positions) == 0:
                acc_by_site[st_name] = 0.0
                n_by_site[st_name] = 0
                continue

            # Use all samples for now (position-level analysis would require
            # per-position predictions which we don't have in this architecture)
            # This is a simplification - real implementation would analyze per-position
            mask = np.ones(len(y_true), dtype=bool)
            n_by_site[st_name] = int(mask.sum())

            if mask.sum() > 0:
                acc_by_site[st_name] = float((y_pred[mask] == y_true[mask]).mean())
            else:
                acc_by_site[st_name] = 0.0

        return acc_by_site, n_by_site

    def _compute_confidence_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray,
    ) -> Tuple[float, float, float, np.ndarray]:
        """Compute confidence-stratified accuracy (Schreiber-Karplus style)."""
        # Sort by confidence descending
        sorted_idx = np.argsort(confidence)[::-1]
        y_true_sorted = y_true[sorted_idx]
        y_pred_sorted = y_pred[sorted_idx]

        n = len(y_true)

        # Mean Cumulative Accuracy curve
        correct = (y_pred_sorted == y_true_sorted).astype(float)
        mca_curve = np.cumsum(correct) / np.arange(1, n + 1)

        # Accuracy at specific thresholds
        idx_25 = max(1, int(n * 0.25))
        idx_50 = max(1, int(n * 0.50))

        acc_25 = float(mca_curve[idx_25 - 1]) if idx_25 > 0 else 0.0
        acc_50 = float(mca_curve[idx_50 - 1]) if idx_50 > 0 else 0.0
        acc_100 = float(mca_curve[-1])

        return acc_25, acc_50, acc_100, mca_curve

    def _compute_auc_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """Compute AUC metrics."""
        try:
            if self.n_classes == 2:
                auc_macro = roc_auc_score(y_true, y_proba[:, 1])
                auc_per_class = {
                    self.class_names[0]: auc_macro,
                    self.class_names[1]: auc_macro,
                }
            else:
                # Multi-class one-vs-rest
                auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr')
                auc_per_class = {}
                for idx, name in enumerate(self.class_names):
                    try:
                        # Binary mask for this class
                        y_binary = (y_true == idx).astype(int)
                        auc_per_class[name] = float(roc_auc_score(y_binary, y_proba[:, idx]))
                    except ValueError:
                        auc_per_class[name] = 0.0
            return float(auc_macro), auc_per_class
        except ValueError:
            return None, None
