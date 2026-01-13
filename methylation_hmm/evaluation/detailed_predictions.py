"""
Detailed prediction export and significance testing for methylation classification.

This module provides:
- export_detailed_predictions: Create comprehensive table with all reads/sites/logits
- compute_significance_values: Add p-values and confidence intervals to metrics
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

# Handle scipy version compatibility
try:
    from scipy.stats import binomtest
    def binom_test(k, n, p):
        return binomtest(k, n, p).pvalue
except ImportError:
    from scipy.stats import binom_test


@dataclass
class SignificanceResults:
    """Statistical significance values for evaluation metrics."""
    # Accuracy significance
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    accuracy_se: float  # Standard error

    # Per-class significance
    per_class_ci: Dict[str, Tuple[float, float]]  # {class: (lower, upper)}

    # Site-type significance
    site_type_ci: Dict[str, Tuple[float, float]]  # {site_type: (lower, upper)}
    site_type_pvalues: Dict[str, float]  # p-value vs overall accuracy

    # Cross-config comparison (if baseline provided)
    vs_baseline_pvalue: Optional[float] = None
    vs_chance_pvalue: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": {
                "ci_95": [self.accuracy_ci_lower, self.accuracy_ci_upper],
                "se": self.accuracy_se,
            },
            "per_class_ci_95": {k: list(v) for k, v in self.per_class_ci.items()},
            "site_type": {
                "ci_95": {k: list(v) for k, v in self.site_type_ci.items()},
                "pvalues_vs_overall": self.site_type_pvalues,
            },
            "vs_baseline_pvalue": self.vs_baseline_pvalue,
            "vs_chance_pvalue": self.vs_chance_pvalue,
        }


def compute_binomial_ci(
    n_correct: int,
    n_total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for accuracy.

    Wilson score is preferred over normal approximation for
    proportions, especially when p is near 0 or 1.
    """
    if n_total == 0:
        return (0.0, 1.0)

    p = n_correct / n_total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    # Wilson score interval
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = (z / denominator) * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))

    return (max(0, center - margin), min(1, center + margin))


def compute_significance_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    site_type_correct: Optional[Dict[str, int]] = None,
    site_type_total: Optional[Dict[str, int]] = None,
    baseline_accuracy: Optional[float] = None,
    confidence_level: float = 0.95,
) -> SignificanceResults:
    """
    Compute significance values for classification results.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        site_type_correct: Dict of correct predictions per site type
        site_type_total: Dict of total predictions per site type
        baseline_accuracy: Accuracy to compare against (e.g., another config)
        confidence_level: Confidence level for intervals (default 0.95)

    Returns:
        SignificanceResults with all computed statistics
    """
    n_total = len(y_true)
    n_correct = int((y_pred == y_true).sum())
    n_classes = len(class_names)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    # Overall accuracy CI
    ci_lower, ci_upper = compute_binomial_ci(n_correct, n_total, confidence_level)
    se = np.sqrt(accuracy * (1 - accuracy) / n_total) if n_total > 0 else 0.0

    # Per-class CI
    per_class_ci = {}
    for idx, name in enumerate(class_names):
        mask = y_true == idx
        class_total = mask.sum()
        class_correct = ((y_pred == idx) & mask).sum()
        per_class_ci[name] = compute_binomial_ci(class_correct, class_total, confidence_level)

    # Site-type CI and p-values
    site_type_ci = {}
    site_type_pvalues = {}

    if site_type_correct and site_type_total:
        for st_name in ['non_cpg', 'cpg', 'homopolymer']:
            st_correct = site_type_correct.get(st_name, 0)
            st_total = site_type_total.get(st_name, 0)

            site_type_ci[st_name] = compute_binomial_ci(st_correct, st_total, confidence_level)

            # Chi-square test: is site-type accuracy different from overall?
            if st_total > 0 and n_total > 0:
                # Expected correct if site-type had same accuracy as overall
                expected_correct = accuracy * st_total
                expected_incorrect = (1 - accuracy) * st_total

                if expected_correct > 5 and expected_incorrect > 5:  # Chi-square validity
                    observed = [st_correct, st_total - st_correct]
                    expected = [expected_correct, expected_incorrect]
                    chi2, pval = stats.chisquare(observed, expected)
                    site_type_pvalues[st_name] = float(pval)
                else:
                    # Use exact binomial test for small samples
                    pval = binom_test(st_correct, st_total, accuracy)
                    site_type_pvalues[st_name] = float(pval)
            else:
                site_type_pvalues[st_name] = 1.0

    # Comparison to baseline (McNemar's test would require paired predictions)
    vs_baseline_pvalue = None
    if baseline_accuracy is not None and n_total > 0:
        # Two-proportion z-test
        p1 = accuracy
        p2 = baseline_accuracy
        pooled = (p1 * n_total + p2 * n_total) / (2 * n_total)
        se_diff = np.sqrt(2 * pooled * (1 - pooled) / n_total)
        if se_diff > 0:
            z = (p1 - p2) / se_diff
            vs_baseline_pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Test vs chance (1/n_classes)
    chance_accuracy = 1 / n_classes
    vs_chance_pvalue = float(binom_test(n_correct, n_total, chance_accuracy))

    return SignificanceResults(
        accuracy_ci_lower=ci_lower,
        accuracy_ci_upper=ci_upper,
        accuracy_se=se,
        per_class_ci=per_class_ci,
        site_type_ci=site_type_ci,
        site_type_pvalues=site_type_pvalues,
        vs_baseline_pvalue=vs_baseline_pvalue,
        vs_chance_pvalue=vs_chance_pvalue,
    )


def export_detailed_predictions(
    results: List,  # List[ClassificationResult]
    test_df: pd.DataFrame,
    site_type_positions: Dict[int, int],
    class_names: List[str],
) -> pd.DataFrame:
    """
    Export comprehensive prediction table with all reads, positions, logits, and site types.

    This table contains one row per (read, cytosine_position) pair, enabling
    flexible downstream analysis and statistical testing.

    Args:
        results: List of ClassificationResult from HMM classification
        test_df: Original test DataFrame with read_id, sample, position, mean_current
        site_type_positions: Dict mapping position -> site_type (0, 1, 2)
        class_names: List of class names ['C', '5mC'] or ['C', '5mC', '5hmC']

    Returns:
        DataFrame with columns:
        - read_id: Unique read identifier
        - true_label: Ground truth modification (C, 5mC, 5hmC)
        - position: Cytosine position (38, 50, 62, 74, 86, 98, 110, 122)
        - site_type: non_cpg, cpg, or homopolymer
        - current_value: Observed signal (pA) at this position
        - log_prob_C: Log-likelihood for C at this position
        - log_prob_5mC: Log-likelihood for 5mC at this position
        - log_prob_5hmC: Log-likelihood for 5hmC (if 3-way mode)
        - position_prediction: Predicted class at this position (argmax)
        - position_correct: Whether position prediction matches true label
        - read_prediction: Overall read-level prediction
        - read_correct: Whether read prediction matches true label
        - confidence: Read-level confidence score
        - read_log_likelihood: Total log-likelihood for the read
    """
    SITE_TYPE_NAMES = {0: 'non_cpg', 1: 'cpg', 2: 'homopolymer'}

    # Get ground truth per read
    read_samples = test_df.groupby('read_id')['sample'].first().to_dict()
    sample_to_label = {'control': 'C', 'C': 'C', '5mC': '5mC', 'mC': '5mC', '5hmC': '5hmC', 'hmC': '5hmC'}

    # Get current values per (read, position)
    current_values = {}
    for _, row in test_df.iterrows():
        key = (row['read_id'], row['position'])
        current_values[key] = row.get('mean_current', np.nan)

    rows = []
    for r in results:
        read_id = r.read_id
        true_sample = read_samples.get(read_id, 'control')
        true_label = sample_to_label.get(true_sample, 'C')

        read_prediction = r.prediction
        read_correct = (read_prediction == true_label)
        confidence = r.confidence
        read_log_likelihood = r.log_likelihood

        # Per-position log-probs
        if r.per_position_log_probs is None:
            continue

        for pos, log_probs in r.per_position_log_probs.items():
            site_type_code = site_type_positions.get(pos, -1)
            site_type = SITE_TYPE_NAMES.get(site_type_code, 'unknown')

            # Position-level prediction
            pos_prediction = max(log_probs, key=log_probs.get)
            pos_correct = (pos_prediction == true_label)

            # Current value at this position
            current = current_values.get((read_id, pos), np.nan)

            row = {
                'read_id': read_id,
                'true_label': true_label,
                'position': pos,
                'site_type': site_type,
                'current_value': current,
                'log_prob_C': log_probs.get('C', np.nan),
                'log_prob_5mC': log_probs.get('5mC', np.nan),
                'position_prediction': pos_prediction,
                'position_correct': pos_correct,
                'read_prediction': read_prediction,
                'read_correct': read_correct,
                'confidence': confidence,
                'read_log_likelihood': read_log_likelihood,
            }

            # Add 5hmC column if in 3-way mode
            if '5hmC' in class_names:
                row['log_prob_5hmC'] = log_probs.get('5hmC', np.nan)

            rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure column order
    base_cols = [
        'read_id', 'true_label', 'position', 'site_type', 'current_value',
        'log_prob_C', 'log_prob_5mC',
    ]
    if '5hmC' in class_names:
        base_cols.append('log_prob_5hmC')
    base_cols.extend([
        'position_prediction', 'position_correct',
        'read_prediction', 'read_correct', 'confidence', 'read_log_likelihood'
    ])

    return df[base_cols]


def compute_site_type_significance_from_table(
    detailed_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Compute significance values from the detailed predictions table.

    Args:
        detailed_df: Output from export_detailed_predictions()

    Returns:
        Dict with per-site-type statistics including CI and p-values
    """
    results = {}

    overall_correct = detailed_df['position_correct'].sum()
    overall_total = len(detailed_df)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0

    for site_type in ['non_cpg', 'cpg', 'homopolymer']:
        site_df = detailed_df[detailed_df['site_type'] == site_type]
        n_total = len(site_df)
        n_correct = site_df['position_correct'].sum()
        accuracy = n_correct / n_total if n_total > 0 else 0

        ci_lower, ci_upper = compute_binomial_ci(n_correct, n_total)

        # Test vs overall accuracy
        if n_total > 0 and overall_total > 0:
            # Exact binomial test
            pval = binom_test(n_correct, n_total, overall_accuracy)
        else:
            pval = 1.0

        results[site_type] = {
            'accuracy': accuracy,
            'n_correct': int(n_correct),
            'n_total': int(n_total),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'pvalue_vs_overall': float(pval),
            'se': np.sqrt(accuracy * (1 - accuracy) / n_total) if n_total > 0 else 0,
        }

    return results


def compare_configurations(
    detailed_dfs: Dict[str, pd.DataFrame],
    baseline_config: str = 'binary_single',
) -> pd.DataFrame:
    """
    Statistical comparison of multiple configurations.

    Args:
        detailed_dfs: Dict mapping config name to detailed predictions DataFrame
        baseline_config: Configuration to use as baseline for comparisons

    Returns:
        DataFrame with comparison statistics
    """
    rows = []

    baseline_df = detailed_dfs.get(baseline_config)
    if baseline_df is not None:
        baseline_accuracy = baseline_df['read_correct'].mean()
    else:
        baseline_accuracy = None

    for config_name, df in detailed_dfs.items():
        n_reads = df['read_id'].nunique()
        n_positions = len(df)

        # Read-level metrics
        read_correct = df.groupby('read_id')['read_correct'].first()
        read_accuracy = read_correct.mean()
        read_ci = compute_binomial_ci(read_correct.sum(), len(read_correct))

        # Position-level metrics
        pos_accuracy = df['position_correct'].mean()
        pos_ci = compute_binomial_ci(df['position_correct'].sum(), len(df))

        # Comparison to baseline
        pval_vs_baseline = None
        if baseline_df is not None and config_name != baseline_config:
            # McNemar's test requires paired data - use approximation
            n = len(read_correct)
            p1, p2 = read_accuracy, baseline_accuracy
            pooled = (p1 + p2) / 2
            se = np.sqrt(2 * pooled * (1 - pooled) / n) if n > 0 else 1
            if se > 0:
                z = (p1 - p2) / se
                pval_vs_baseline = float(2 * (1 - stats.norm.cdf(abs(z))))

        rows.append({
            'configuration': config_name,
            'n_reads': n_reads,
            'n_positions': n_positions,
            'read_accuracy': read_accuracy,
            'read_ci_lower': read_ci[0],
            'read_ci_upper': read_ci[1],
            'position_accuracy': pos_accuracy,
            'position_ci_lower': pos_ci[0],
            'position_ci_upper': pos_ci[1],
            'pvalue_vs_baseline': pval_vs_baseline,
        })

    return pd.DataFrame(rows)
