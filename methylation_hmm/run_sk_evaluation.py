#!/usr/bin/env python
"""
Run Schreiber-Karplus style evaluation on both classifiers.

Compares:
1. Simplified (Two-HMM) classifier
2. Fork HMM classifier

Generates metrics matching the original paper's methodology.
"""

import sys
sys.path.insert(0, '/home/jesse/repos/bio_hackathon')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier
from methylation_hmm.schreiber_karplus_evaluation import (
    SchreiberKarplusEvaluator,
    SchreiberKarplusMetrics,
    print_schreiber_karplus_report,
    generate_mca_plot_data,
    compare_to_paper_baseline
)


def evaluate_simplified_classifier(df: pd.DataFrame) -> SchreiberKarplusMetrics:
    """Evaluate the two-HMM likelihood ratio classifier."""
    print("\n" + "=" * 60)
    print("EVALUATING: Simplified (Two-HMM) Classifier")
    print("=" * 60)

    # Train/test split
    POSITIONS = ['38', '50', '62', '74', '86', '98', '110', '122']
    df_clean = df.dropna(subset=POSITIONS)

    control = df_clean[df_clean['sample'] == 'control']
    modified = df_clean[df_clean['sample'] == 'modified']

    test_split = 0.2
    n_control_train = int(len(control) * (1 - test_split))
    n_modified_train = int(len(modified) * (1 - test_split))

    train_control = control.iloc[:n_control_train]
    train_modified = modified.iloc[:n_modified_train]
    test_df = pd.concat([
        control.iloc[n_control_train:],
        modified.iloc[n_modified_train:]
    ])

    # Train classifier
    import torch
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

    print(f"Training: {len(train_control)} control + {len(train_modified)} modified")
    print(f"Testing: {len(test_df)} reads")

    # Evaluate
    evaluator = SchreiberKarplusEvaluator(classifier)
    metrics = evaluator.compute_metrics(test_df)

    return metrics, evaluator


def evaluate_fork_hmm(df: pd.DataFrame) -> SchreiberKarplusMetrics:
    """Evaluate the fork-based single HMM classifier."""
    print("\n" + "=" * 60)
    print("EVALUATING: Fork HMM (Single Model) Classifier")
    print("=" * 60)

    try:
        from methylation_hmm.fork_hmm import ForkHMMClassifier
    except ImportError:
        print("Fork HMM classifier not available, skipping...")
        return None, None

    POSITIONS = ['38', '50', '62', '74', '86', '98', '110', '122']
    df_clean = df.dropna(subset=POSITIONS)

    control = df_clean[df_clean['sample'] == 'control']
    modified = df_clean[df_clean['sample'] == 'modified']

    test_split = 0.2
    n_control_train = int(len(control) * (1 - test_split))
    n_modified_train = int(len(modified) * (1 - test_split))

    train_control = control.iloc[:n_control_train]
    train_modified = modified.iloc[:n_modified_train]
    train_df = pd.concat([train_control, train_modified])

    test_df = pd.concat([
        control.iloc[n_control_train:],
        modified.iloc[n_modified_train:]
    ])

    # Create and train fork HMM classifier
    classifier = ForkHMMClassifier()
    classifier._compute_emission_params(train_control, train_modified)
    classifier._build_fork_hmm()

    X_train = train_df[POSITIONS].values.astype(np.float32)
    classifier._train(X_train)

    print(f"Training: {len(train_df)} reads")
    print(f"Testing: {len(test_df)} reads")

    # Create adapter to match SimplifiedMethylationClassifier interface
    class ForkHMMAdapter:
        def __init__(self, fork_classifier):
            self.classifier = fork_classifier
            self.POSITIONS = POSITIONS

        def predict_proba(self, X):
            # Use the fork HMM's path likelihood computation
            logp_c, logp_mc = self.classifier._compute_path_likelihoods(X)
            return logp_c, logp_mc

    adapter = ForkHMMAdapter(classifier)
    evaluator = SchreiberKarplusEvaluator(adapter)
    metrics = evaluator.compute_metrics(test_df)

    return metrics, evaluator


def plot_mca_comparison(
    simplified_metrics: SchreiberKarplusMetrics,
    fork_metrics: SchreiberKarplusMetrics,
    output_path: str
):
    """Generate MCA curve comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Paper baseline
    baseline = compare_to_paper_baseline()

    # Left plot: MCA curves
    ax1 = axes[0]
    n1 = len(simplified_metrics.mca_curve)
    n2 = len(fork_metrics.mca_curve) if fork_metrics else 0

    coverage1 = np.linspace(0, 1, n1)
    ax1.plot(coverage1 * 100, simplified_metrics.mca_curve * 100,
             'b-', linewidth=2, label='Simplified (Two-HMM)')

    if fork_metrics and n2 > 0:
        coverage2 = np.linspace(0, 1, n2)
        ax1.plot(coverage2 * 100, fork_metrics.mca_curve * 100,
                 'r-', linewidth=2, label='Fork HMM')

    # Paper reference line
    ax1.axhline(y=baseline['accuracy_at_26pct'] * 100, color='g', linestyle='--',
                alpha=0.7, label=f"Paper @ 26%: {baseline['accuracy_at_26pct']:.1%}")
    ax1.axvline(x=26, color='gray', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Coverage (%)', fontsize=12)
    ax1.set_ylabel('Mean Cumulative Accuracy (%)', fontsize=12)
    ax1.set_title('MCA Curve (Analog to MCSC)', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(50, 100)

    # Right plot: Confidence distribution
    ax2 = axes[1]
    ax2.hist(simplified_metrics.confidence_scores, bins=50, alpha=0.6,
             label='Simplified', color='blue', density=True)

    if fork_metrics and len(fork_metrics.confidence_scores) > 0:
        ax2.hist(fork_metrics.confidence_scores, bins=50, alpha=0.6,
                 label='Fork HMM', color='red', density=True)

    ax2.set_xlabel('Confidence Score (|log ratio|)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Confidence Distribution (Analog to Filter Score)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_comparison_table(
    simplified: SchreiberKarplusMetrics,
    fork: SchreiberKarplusMetrics
) -> str:
    """Generate markdown comparison table."""
    baseline = compare_to_paper_baseline()

    lines = [
        "## Schreiber-Karplus Style Evaluation Results",
        "",
        "### Methodology Mapping",
        "",
        "| Original Paper Metric | Our Analog | Description |",
        "|----------------------|------------|-------------|",
        "| Filter Score | abs(log ratio) | Confidence that read is classifiable |",
        "| Soft Call | 0/1 correctness | Per-read classification accuracy |",
        "| MCSC | MCA (Mean Cumulative Accuracy) | Accuracy at confidence threshold |",
        "| On-pathway events | High-confidence reads | Reads amenable to classification |",
        "",
        "### Key Results",
        "",
        "| Metric | Paper Baseline | Simplified | Fork HMM |",
        "|--------|----------------|------------|----------|",
        f"| **Accuracy @ 25%** | **97.7%** | **{simplified.accuracy_at_25pct:.1%}** | {fork.accuracy_at_25pct:.1%} |" if fork else f"| **Accuracy @ 25%** | **97.7%** | **{simplified.accuracy_at_25pct:.1%}** | N/A |",
        f"| Accuracy @ 50% | ~90% | {simplified.accuracy_at_50pct:.1%} | {fork.accuracy_at_50pct:.1%} |" if fork else f"| Accuracy @ 50% | ~90% | {simplified.accuracy_at_50pct:.1%} | N/A |",
        f"| Overall Accuracy | ~75-80% | {simplified.overall_accuracy:.1%} | {fork.overall_accuracy:.1%} |" if fork else f"| Overall Accuracy | ~75-80% | {simplified.overall_accuracy:.1%} | N/A |",
        f"| ROC AUC | N/A | {simplified.roc_auc:.3f} | {fork.roc_auc:.3f} |" if fork else f"| ROC AUC | N/A | {simplified.roc_auc:.3f} | N/A |",
        "",
        "### Per-Class Performance",
        "",
        "| Metric | Simplified | Fork HMM |",
        "|--------|------------|----------|",
        f"| Control Accuracy | {simplified.control_accuracy:.1%} | {fork.control_accuracy:.1%} |" if fork else f"| Control Accuracy | {simplified.control_accuracy:.1%} | N/A |",
        f"| Modified Accuracy | {simplified.modified_accuracy:.1%} | {fork.modified_accuracy:.1%} |" if fork else f"| Modified Accuracy | {simplified.modified_accuracy:.1%} | N/A |",
        "",
        "### Interpretation",
        "",
        "The paper achieves 97.7% accuracy on the top 26% most confident predictions.",
        f"Our simplified classifier achieves **{simplified.accuracy_at_25pct:.1%}** on the top 25%.",
        "",
        "**Gap Analysis:**",
        f"- Paper top-25% accuracy: 97.7%",
        f"- Our top-25% accuracy: {simplified.accuracy_at_25pct:.1%}",
        f"- Gap: {(baseline['accuracy_at_26pct'] - simplified.accuracy_at_25pct) * 100:.1f} percentage points",
        "",
        "**Possible reasons for the gap:**",
        "1. Different chemistry (R10.4.1 vs older R7.3)",
        "2. Different reference constructs (synthetic 5mers vs phi29 polymerase)",
        "3. No independent label validation fork (T/X/CAT in paper)",
        "4. Simpler model architecture (8 states vs ~300+ states)",
    ]

    return "\n".join(lines)


def main():
    """Run full Schreiber-Karplus evaluation."""
    # Create output directory
    output_dir = Path('output/plots')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv('output/hmm_training_sequences.csv')
    print(f"Loaded {len(df)} reads")

    # Evaluate both classifiers
    simplified_metrics, simplified_eval = evaluate_simplified_classifier(df)
    print_schreiber_karplus_report(simplified_metrics)

    fork_metrics, fork_eval = evaluate_fork_hmm(df)
    if fork_metrics:
        print_schreiber_karplus_report(fork_metrics)

    # Generate plots
    plot_mca_comparison(
        simplified_metrics,
        fork_metrics,
        str(output_dir / 'schreiber_karplus_mca_comparison.png')
    )

    # Generate comparison table
    comparison_md = generate_comparison_table(simplified_metrics, fork_metrics)
    print("\n" + comparison_md)

    # Save results
    with open('output/schreiber_karplus_evaluation.md', 'w') as f:
        f.write(comparison_md)
    print("\nSaved: output/schreiber_karplus_evaluation.md")

    # Save MCA curve data
    mca_data = generate_mca_plot_data(simplified_metrics)
    mca_data.to_csv('output/simplified_mca_curve.csv', index=False)
    print("Saved: output/simplified_mca_curve.csv")

    if fork_metrics:
        fork_mca_data = generate_mca_plot_data(fork_metrics)
        fork_mca_data.to_csv('output/fork_hmm_mca_curve.csv', index=False)
        print("Saved: output/fork_hmm_mca_curve.csv")


if __name__ == "__main__":
    main()
