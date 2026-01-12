"""
Compare Simplified vs Fork HMM classifiers.

This script runs both classifiers on the same data and generates
comparison plots and metrics.
"""

import sys
sys.path.insert(0, '/home/jesse/repos/bio_hackathon')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier
from methylation_hmm.fork_hmm import ForkHMMClassifier, cross_validate


def load_and_split_data(training_csv: str, test_split: float = 0.2):
    """Load data and create train/test split."""
    df = pd.read_csv(training_csv)
    positions = ['38', '50', '62', '74', '86', '98', '110', '122']
    df_clean = df.dropna(subset=positions)

    control = df_clean[df_clean['sample'] == 'control']
    modified = df_clean[df_clean['sample'] == 'modified']

    n_ctrl_train = int(len(control) * (1 - test_split))
    n_mod_train = int(len(modified) * (1 - test_split))

    train_df = pd.concat([
        control.iloc[:n_ctrl_train],
        modified.iloc[:n_mod_train]
    ])
    test_df = pd.concat([
        control.iloc[n_ctrl_train:],
        modified.iloc[n_mod_train:]
    ])

    return train_df, test_df, positions


def run_simplified_classifier(train_df, test_df, positions):
    """Train and evaluate simplified (two-HMM) classifier."""
    ctrl_train = train_df[train_df['sample'] == 'control']
    mod_train = train_df[train_df['sample'] == 'modified']

    classifier = SimplifiedMethylationClassifier()
    classifier._compute_emission_params(ctrl_train, mod_train)
    classifier._build_models_from_params()

    # Train
    X_ctrl = ctrl_train[positions].values.astype(np.float32)
    X_mod = mod_train[positions].values.astype(np.float32)
    classifier.fit(X_ctrl, X_mod)

    # Evaluate
    X_test = test_df[positions].values.astype(np.float32)
    y_true = (test_df['sample'].values == 'modified').astype(int)

    logp_ctrl, logp_mod = classifier.predict_proba(X_test)
    scores = logp_mod - logp_ctrl
    y_pred = (scores > 0).astype(int)

    accuracy = (y_pred == y_true).mean()
    auc_score = roc_auc_score(y_true, scores)

    return {
        'name': 'Simplified (Two-HMM)',
        'accuracy': accuracy,
        'auc': auc_score,
        'y_true': y_true,
        'y_pred': y_pred,
        'scores': scores
    }


def run_fork_classifier(train_df, test_df, positions):
    """Train and evaluate fork HMM classifier."""
    ctrl_train = train_df[train_df['sample'] == 'control']
    mod_train = train_df[train_df['sample'] == 'modified']

    classifier = ForkHMMClassifier()
    classifier._compute_emission_params(ctrl_train, mod_train)
    classifier._build_fork_hmm()

    # Train on all data
    X_train = train_df[positions].values.astype(np.float32)
    classifier._train(X_train)

    # Evaluate
    metrics = classifier.evaluate(test_df)

    return {
        'name': 'Fork HMM (Single Model)',
        'accuracy': metrics['accuracy'],
        'auc': metrics['auc'],
        'y_true': metrics['y_true'],
        'y_pred': metrics['y_pred'],
        'scores': metrics['scores']
    }


def cross_validate_both(training_csv: str, n_folds: int = 5):
    """Run cross-validation for both classifiers."""
    df = pd.read_csv(training_csv)
    positions = ['38', '50', '62', '74', '86', '98', '110', '122']
    df_clean = df.dropna(subset=positions)

    X = df_clean[positions].values.astype(np.float32)
    y = (df_clean['sample'].values == 'modified').astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    simplified_accs = []
    simplified_aucs = []
    fork_accs = []
    fork_aucs = []

    print(f"\n{'='*60}")
    print(f"{n_folds}-FOLD CROSS-VALIDATION COMPARISON")
    print(f"{'='*60}\n")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_df = df_clean.iloc[train_idx]
        test_df = df_clean.iloc[test_idx]

        # Simplified classifier
        simp_results = run_simplified_classifier(train_df, test_df, positions)
        simplified_accs.append(simp_results['accuracy'])
        simplified_aucs.append(simp_results['auc'])

        # Fork classifier
        fork_results = run_fork_classifier(train_df, test_df, positions)
        fork_accs.append(fork_results['accuracy'])
        fork_aucs.append(fork_results['auc'])

        print(f"Fold {fold+1}:")
        print(f"  Simplified: Acc={simp_results['accuracy']:.3f}, AUC={simp_results['auc']:.3f}")
        print(f"  Fork HMM:   Acc={fork_results['accuracy']:.3f}, AUC={fork_results['auc']:.3f}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nSimplified Classifier:")
    print(f"  Accuracy: {np.mean(simplified_accs):.3f} +/- {np.std(simplified_accs):.3f}")
    print(f"  AUC:      {np.mean(simplified_aucs):.3f} +/- {np.std(simplified_aucs):.3f}")
    print(f"\nFork HMM Classifier:")
    print(f"  Accuracy: {np.mean(fork_accs):.3f} +/- {np.std(fork_accs):.3f}")
    print(f"  AUC:      {np.mean(fork_aucs):.3f} +/- {np.std(fork_aucs):.3f}")

    return {
        'simplified': {'accs': simplified_accs, 'aucs': simplified_aucs},
        'fork': {'accs': fork_accs, 'aucs': fork_aucs}
    }


def plot_comparison(simp_results, fork_results, output_dir: str):
    """Generate comparison plots."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # ROC Comparison
    plt.figure(figsize=(10, 8))

    fpr_s, tpr_s, _ = roc_curve(simp_results['y_true'], simp_results['scores'])
    auc_s = auc(fpr_s, tpr_s)
    plt.plot(fpr_s, tpr_s, 'b-', lw=2,
             label=f'Simplified (AUC = {auc_s:.3f})')

    fpr_f, tpr_f, _ = roc_curve(fork_results['y_true'], fork_results['scores'])
    auc_f = auc(fpr_f, tpr_f)
    plt.plot(fpr_f, tpr_f, 'r-', lw=2,
             label=f'Fork HMM (AUC = {auc_f:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Comparison: Simplified vs Fork HMM', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'classifier_comparison_roc.png', dpi=150)
    plt.close()

    # Accuracy by confidence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (results, color, name) in zip(
        axes,
        [(simp_results, 'blue', 'Simplified'),
         (fork_results, 'red', 'Fork HMM')]
    ):
        y_true = results['y_true']
        y_pred = results['y_pred']
        scores = results['scores']
        confidence = np.abs(scores)

        sorted_idx = np.argsort(-confidence)
        y_true_s = y_true[sorted_idx]
        y_pred_s = y_pred[sorted_idx]

        coverages = []
        accuracies = []

        for pct in np.linspace(0.05, 1.0, 20):
            n = int(len(y_true) * pct)
            if n < 10:
                continue
            acc = (y_true_s[:n] == y_pred_s[:n]).mean()
            coverages.append(pct * 100)
            accuracies.append(acc * 100)

        ax.plot(coverages, accuracies, f'{color[0]}-o', lw=2, markersize=6)
        ax.set_xlabel('Coverage (%)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{name}: Accuracy vs Coverage', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 105])
        ax.set_ylim([50, 100])

        # Mark 26%
        idx_26 = np.argmin(np.abs(np.array(coverages) - 26))
        if idx_26 < len(accuracies):
            ax.axvline(x=26, color='gray', linestyle='--', alpha=0.7)
            ax.annotate(f'Top 26%: {accuracies[idx_26]:.1f}%',
                       xy=(26, accuracies[idx_26]),
                       xytext=(40, accuracies[idx_26]),
                       fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / 'classifier_comparison_accuracy.png', dpi=150)
    plt.close()

    # Score distributions comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (results, name) in zip(
        axes,
        [(simp_results, 'Simplified'), (fork_results, 'Fork HMM')]
    ):
        y_true = results['y_true']
        scores = results['scores']

        bins = np.linspace(scores.min(), scores.max(), 40)
        ax.hist(scores[y_true == 0], bins=bins, alpha=0.6,
                label=f'Control (n={sum(y_true==0)})', color='blue', density=True)
        ax.hist(scores[y_true == 1], bins=bins, alpha=0.6,
                label=f'Modified (n={sum(y_true==1)})', color='red', density=True)
        ax.axvline(x=0, color='black', linestyle='--', lw=2)
        ax.set_xlabel('Log-Likelihood Ratio', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{name}: Score Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'classifier_comparison_scores.png', dpi=150)
    plt.close()


def main():
    training_csv = '/home/jesse/repos/bio_hackathon/output/hmm_training_sequences.csv'
    output_dir = '/home/jesse/repos/bio_hackathon/output/plots'

    print("="*60)
    print("METHYLATION CLASSIFIER COMPARISON")
    print("="*60)

    # Load data
    train_df, test_df, positions = load_and_split_data(training_csv)
    print(f"\nTraining: {len(train_df)} reads")
    print(f"Test: {len(test_df)} reads")

    # Train and evaluate both classifiers
    print("\n" + "-"*40)
    print("SINGLE TRAIN/TEST SPLIT")
    print("-"*40)

    simp_results = run_simplified_classifier(train_df, test_df, positions)
    print(f"\nSimplified Classifier:")
    print(f"  Accuracy: {simp_results['accuracy']:.3f}")
    print(f"  AUC: {simp_results['auc']:.3f}")

    fork_results = run_fork_classifier(train_df, test_df, positions)
    print(f"\nFork HMM Classifier:")
    print(f"  Accuracy: {fork_results['accuracy']:.3f}")
    print(f"  AUC: {fork_results['auc']:.3f}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_comparison(simp_results, fork_results, output_dir)

    # Cross-validation
    print("\n" + "-"*40)
    print("5-FOLD CROSS-VALIDATION")
    print("-"*40)
    cv_results = cross_validate_both(training_csv, n_folds=5)

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print("\nSimplified (Two-HMM Likelihood Ratio) Classifier:")
    print(f"  Test Accuracy: {simp_results['accuracy']:.1%}")
    print(f"  Test AUC: {simp_results['auc']:.3f}")
    print(f"  CV Accuracy: {np.mean(cv_results['simplified']['accs']):.1%} +/- {np.std(cv_results['simplified']['accs']):.1%}")
    print(f"  CV AUC: {np.mean(cv_results['simplified']['aucs']):.3f} +/- {np.std(cv_results['simplified']['aucs']):.3f}")

    print("\nFork HMM (Single Model with Forks) Classifier:")
    print(f"  Test Accuracy: {fork_results['accuracy']:.1%}")
    print(f"  Test AUC: {fork_results['auc']:.3f}")
    print(f"  CV Accuracy: {np.mean(cv_results['fork']['accs']):.1%} +/- {np.std(cv_results['fork']['accs']):.1%}")
    print(f"  CV AUC: {np.mean(cv_results['fork']['aucs']):.3f} +/- {np.std(cv_results['fork']['aucs']):.3f}")

    print(f"\nPlots saved to: {output_dir}")


if __name__ == '__main__':
    main()
