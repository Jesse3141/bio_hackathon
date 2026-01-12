"""
Generate AUC/ROC plots for the methylation classifier.

Creates:
1. ROC curve with AUC score
2. Precision-Recall curve
3. Accuracy vs confidence threshold curve
4. Per-position ROC curves
"""

import sys
sys.path.insert(0, '/home/jesse/repos/bio_hackathon')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from pathlib import Path

from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier


def load_data(training_csv: str):
    """Load and split training data."""
    df = pd.read_csv(training_csv)
    positions = ['38', '50', '62', '74', '86', '98', '110', '122']
    df_clean = df.dropna(subset=positions)

    # 80/20 train/test split
    control = df_clean[df_clean['sample'] == 'control']
    modified = df_clean[df_clean['sample'] == 'modified']

    n_ctrl_train = int(len(control) * 0.8)
    n_mod_train = int(len(modified) * 0.8)

    train_df = pd.concat([
        control.iloc[:n_ctrl_train],
        modified.iloc[:n_mod_train]
    ])
    test_df = pd.concat([
        control.iloc[n_ctrl_train:],
        modified.iloc[n_mod_train:]
    ])

    return train_df, test_df, positions


def plot_roc_curve(classifier, test_df, positions, output_dir: Path):
    """Generate ROC curve."""
    X = test_df[positions].values.astype(np.float32)
    y_true = (test_df['sample'].values == 'modified').astype(int)

    logp_ctrl, logp_mod = classifier.predict_proba(X)

    # Use log ratio as the score (higher = more likely modified)
    scores = logp_mod - logp_ctrl

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Methylation Classification (C vs 5mC)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150)
    plt.close()

    print(f"ROC AUC: {roc_auc:.4f}")
    return roc_auc


def plot_precision_recall(classifier, test_df, positions, output_dir: Path):
    """Generate Precision-Recall curve."""
    X = test_df[positions].values.astype(np.float32)
    y_true = (test_df['sample'].values == 'modified').astype(int)

    logp_ctrl, logp_mod = classifier.predict_proba(X)
    scores = logp_mod - logp_ctrl

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AP = {ap:.3f})')

    # Baseline (proportion of positives)
    baseline = y_true.mean()
    plt.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                label=f'Random baseline ({baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve: Methylation Classification', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=150)
    plt.close()

    print(f"Average Precision: {ap:.4f}")
    return ap


def plot_accuracy_vs_confidence(classifier, test_df, positions, output_dir: Path):
    """
    Plot accuracy at different confidence thresholds.

    Replicates the paper's analysis showing accuracy improves when
    filtering to high-confidence predictions.
    """
    X = test_df[positions].values.astype(np.float32)
    y_true = (test_df['sample'].values == 'modified').astype(int)

    logp_ctrl, logp_mod = classifier.predict_proba(X)
    log_ratio = logp_mod - logp_ctrl
    confidence = np.abs(log_ratio)
    y_pred = (log_ratio > 0).astype(int)

    # Sort by confidence descending
    sorted_idx = np.argsort(-confidence)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    coverages = []
    accuracies = []

    for pct in np.linspace(0.05, 1.0, 20):
        n_keep = int(len(y_true) * pct)
        if n_keep < 10:
            continue
        acc = (y_true_sorted[:n_keep] == y_pred_sorted[:n_keep]).mean()
        coverages.append(pct)
        accuracies.append(acc)

    plt.figure(figsize=(8, 6))
    plt.plot(np.array(coverages) * 100, np.array(accuracies) * 100,
             'b-o', lw=2, markersize=6)
    plt.xlabel('Coverage (% of reads)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Coverage (Higher Confidence = Higher Accuracy)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 105])
    plt.ylim([50, 100])

    # Add annotation for top 26% (paper's threshold)
    idx_26 = np.argmin(np.abs(np.array(coverages) - 0.26))
    if idx_26 < len(accuracies):
        plt.axvline(x=26, color='r', linestyle='--', alpha=0.7)
        plt.annotate(f'Top 26%: {accuracies[idx_26]*100:.1f}%',
                    xy=(26, accuracies[idx_26]*100),
                    xytext=(35, accuracies[idx_26]*100 - 5),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_confidence.png', dpi=150)
    plt.close()

    print(f"Overall accuracy: {accuracies[-1]*100:.1f}%")
    if idx_26 < len(accuracies):
        print(f"Top 26% accuracy: {accuracies[idx_26]*100:.1f}%")


def plot_per_position_roc(classifier, test_df, positions, output_dir: Path):
    """Generate ROC curves for each position separately."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    aucs = {}

    for i, pos in enumerate(positions):
        # Get data for this position only
        pos_data = test_df[[pos, 'sample']].dropna()
        X = pos_data[pos].values.reshape(-1, 1).astype(np.float32)
        y_true = (pos_data['sample'].values == 'modified').astype(int)

        if len(y_true) < 10:
            continue

        # Simple single-position score: distance from control mean
        ctrl_mean = X[y_true == 0].mean()
        mod_mean = X[y_true == 1].mean()

        # Score: how much signal differs from control mean (positive = toward modified)
        if mod_mean > ctrl_mean:
            scores = X.flatten()  # Higher signal = more modified
        else:
            scores = -X.flatten()  # Lower signal = more modified

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        aucs[pos] = roc_auc

        axes[i].plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'AUC = {roc_auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('FPR')
        axes[i].set_ylabel('TPR')
        axes[i].set_title(f'Position {pos}')
        axes[i].legend(loc='lower right')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Per-Position ROC Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_position_roc.png', dpi=150)
    plt.close()

    print("\nPer-position AUCs:")
    for pos, auc_val in aucs.items():
        print(f"  Position {pos}: {auc_val:.3f}")


def plot_score_distributions(classifier, test_df, positions, output_dir: Path):
    """Plot distribution of log-likelihood ratios for each class."""
    X = test_df[positions].values.astype(np.float32)
    y_true = (test_df['sample'].values == 'modified').astype(int)

    logp_ctrl, logp_mod = classifier.predict_proba(X)
    log_ratio = logp_mod - logp_ctrl

    plt.figure(figsize=(10, 6))

    bins = np.linspace(log_ratio.min(), log_ratio.max(), 50)

    plt.hist(log_ratio[y_true == 0], bins=bins, alpha=0.6,
             label=f'Control (C) n={sum(y_true==0)}', color='blue', density=True)
    plt.hist(log_ratio[y_true == 1], bins=bins, alpha=0.6,
             label=f'Modified (5mC) n={sum(y_true==1)}', color='red', density=True)

    plt.axvline(x=0, color='black', linestyle='--', lw=2, label='Decision boundary')

    plt.xlabel('Log-Likelihood Ratio: log P(5mC) - log P(C)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Score Distribution by True Class', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distributions.png', dpi=150)
    plt.close()


def main():
    training_csv = '/home/jesse/repos/bio_hackathon/output/hmm_training_sequences.csv'
    output_dir = Path('/home/jesse/repos/bio_hackathon/output/plots')
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    train_df, test_df, positions = load_data(training_csv)
    print(f"Training: {len(train_df)} reads")
    print(f"Test: {len(test_df)} reads")

    print("\nTraining simplified classifier...")
    classifier, _ = SimplifiedMethylationClassifier.from_training_data(training_csv)

    print("\nGenerating plots...")

    print("\n1. ROC Curve")
    roc_auc = plot_roc_curve(classifier, test_df, positions, output_dir)

    print("\n2. Precision-Recall Curve")
    ap = plot_precision_recall(classifier, test_df, positions, output_dir)

    print("\n3. Accuracy vs Confidence")
    plot_accuracy_vs_confidence(classifier, test_df, positions, output_dir)

    print("\n4. Per-Position ROC Curves")
    plot_per_position_roc(classifier, test_df, positions, output_dir)

    print("\n5. Score Distributions")
    plot_score_distributions(classifier, test_df, positions, output_dir)

    print(f"\nAll plots saved to: {output_dir}")

    # Summary
    print("\n" + "="*50)
    print("SIMPLIFIED CLASSIFIER SUMMARY")
    print("="*50)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")


if __name__ == '__main__':
    main()
