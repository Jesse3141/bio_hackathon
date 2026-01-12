"""
Output formatters for evaluation results.

Provides functions to save metrics to JSON, CSV, and generate plots.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

from .framework import UnifiedMetrics


def save_json(
    metrics: UnifiedMetrics,
    output_path: Union[str, Path],
    include_mca_curve: bool = False,
) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: UnifiedMetrics object
        output_path: Path to output JSON file
        include_mca_curve: Whether to include full MCA curve array
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = metrics.to_dict()
    data['timestamp'] = datetime.now().isoformat()

    if include_mca_curve and metrics.mca_curve is not None:
        data['mca_curve'] = metrics.mca_curve.tolist()

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def save_csv(
    metrics: UnifiedMetrics,
    output_path: Union[str, Path],
) -> None:
    """
    Save metrics to CSV file in flat format.

    Args:
        metrics: UnifiedMetrics object
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    base = {
        'model': metrics.model_name,
        'mode': metrics.mode,
        'adapters': ','.join(metrics.adapters),
    }

    # Overall metrics
    rows.append({**base, 'metric_type': 'overall', 'metric_name': 'accuracy',
                 'value': metrics.overall_accuracy})
    rows.append({**base, 'metric_type': 'overall', 'metric_name': 'n_samples',
                 'value': metrics.n_samples})

    # Per-class accuracy
    for class_name, acc in metrics.per_class_accuracy.items():
        rows.append({**base, 'metric_type': 'per_class',
                     'metric_name': f'{class_name}_accuracy', 'value': acc})

    # Site type accuracy
    for site_type, acc in metrics.accuracy_by_site_type.items():
        rows.append({**base, 'metric_type': 'site_type',
                     'metric_name': f'{site_type}_accuracy', 'value': acc})

    # Confidence-stratified accuracy
    rows.append({**base, 'metric_type': 'confidence',
                 'metric_name': 'top_25pct_accuracy', 'value': metrics.accuracy_at_top_25pct})
    rows.append({**base, 'metric_type': 'confidence',
                 'metric_name': 'top_50pct_accuracy', 'value': metrics.accuracy_at_top_50pct})
    rows.append({**base, 'metric_type': 'confidence',
                 'metric_name': 'top_100pct_accuracy', 'value': metrics.accuracy_at_100pct})

    # AUC
    if metrics.auc_macro is not None:
        rows.append({**base, 'metric_type': 'auc',
                     'metric_name': 'macro', 'value': metrics.auc_macro})

    # Cross-validation
    if metrics.cv_mean_accuracy is not None:
        rows.append({**base, 'metric_type': 'cv',
                     'metric_name': 'mean_accuracy', 'value': metrics.cv_mean_accuracy})
        rows.append({**base, 'metric_type': 'cv',
                     'metric_name': 'std_accuracy', 'value': metrics.cv_std_accuracy})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def generate_plots(
    metrics: UnifiedMetrics,
    output_dir: Union[str, Path],
    figsize: tuple = (10, 6),
) -> List[str]:
    """
    Generate evaluation plots.

    Args:
        metrics: UnifiedMetrics object
        output_dir: Directory for output plots
        figsize: Figure size (width, height)

    Returns:
        List of generated plot file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping plots.")
        return generated_files

    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=figsize)
    n_classes = len(metrics.per_class_accuracy)
    class_names = list(metrics.per_class_accuracy.keys())

    sns.heatmap(
        metrics.confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {metrics.model_name} ({metrics.mode})')

    conf_path = output_dir / 'confusion_matrix.png'
    plt.savefig(conf_path, dpi=150, bbox_inches='tight')
    plt.close()
    generated_files.append(str(conf_path))

    # 2. MCA Curve
    if metrics.mca_curve is not None and len(metrics.mca_curve) > 0:
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(1, len(metrics.mca_curve) + 1) / len(metrics.mca_curve) * 100
        ax.plot(x, metrics.mca_curve * 100, 'b-', linewidth=2)

        # Mark key points
        ax.axhline(y=metrics.accuracy_at_top_25pct * 100, color='r', linestyle='--',
                   alpha=0.5, label=f'Top 25%: {metrics.accuracy_at_top_25pct:.1%}')
        ax.axhline(y=metrics.accuracy_at_top_50pct * 100, color='orange', linestyle='--',
                   alpha=0.5, label=f'Top 50%: {metrics.accuracy_at_top_50pct:.1%}')
        ax.axhline(y=metrics.accuracy_at_100pct * 100, color='green', linestyle='--',
                   alpha=0.5, label=f'All: {metrics.accuracy_at_100pct:.1%}')

        ax.set_xlabel('Coverage (%)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Mean Cumulative Accuracy - {metrics.model_name}')
        ax.legend(loc='lower left')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        mca_path = output_dir / 'mca_curve.png'
        plt.savefig(mca_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_files.append(str(mca_path))

    # 3. Accuracy by Site Type
    if metrics.accuracy_by_site_type:
        fig, ax = plt.subplots(figsize=(8, 5))

        site_types = list(metrics.accuracy_by_site_type.keys())
        accuracies = [metrics.accuracy_by_site_type[st] * 100 for st in site_types]

        bars = ax.bar(site_types, accuracies, color=['steelblue', 'coral', 'mediumseagreen'])

        ax.set_xlabel('Site Type')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy by Site Type - {metrics.model_name}')
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        site_path = output_dir / 'accuracy_by_site_type.png'
        plt.savefig(site_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_files.append(str(site_path))

    # 4. Per-class accuracy
    if metrics.per_class_accuracy:
        fig, ax = plt.subplots(figsize=(8, 5))

        classes = list(metrics.per_class_accuracy.keys())
        accuracies = [metrics.per_class_accuracy[c] * 100 for c in classes]

        colors = ['#2ecc71', '#e74c3c', '#3498db'][:len(classes)]
        bars = ax.bar(classes, accuracies, color=colors)

        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Per-Class Accuracy - {metrics.model_name}')
        ax.set_ylim(0, 100)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        class_path = output_dir / 'per_class_accuracy.png'
        plt.savefig(class_path, dpi=150, bbox_inches='tight')
        plt.close()
        generated_files.append(str(class_path))

    return generated_files


def generate_comparison_table(
    all_metrics: Dict[str, UnifiedMetrics],
    output_path: Union[str, Path],
) -> None:
    """
    Generate a markdown comparison table for multiple models.

    Args:
        all_metrics: Dict mapping model_name -> UnifiedMetrics
        output_path: Path to output markdown file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Model Comparison")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Mode | Overall Acc | Top 25% Acc | Top 50% Acc | AUC |")
    lines.append("|-------|------|-------------|-------------|-------------|-----|")

    for name, metrics in all_metrics.items():
        auc_str = f"{metrics.auc_macro:.3f}" if metrics.auc_macro else "N/A"
        lines.append(
            f"| {metrics.model_name} | {metrics.mode} | "
            f"{metrics.overall_accuracy:.1%} | "
            f"{metrics.accuracy_at_top_25pct:.1%} | "
            f"{metrics.accuracy_at_top_50pct:.1%} | "
            f"{auc_str} |"
        )

    lines.append("")

    # Cross-validation table (if available)
    has_cv = any(m.cv_mean_accuracy is not None for m in all_metrics.values())
    if has_cv:
        lines.append("## Cross-Validation Results")
        lines.append("")
        lines.append("| Model | Mode | CV Accuracy | CV AUC |")
        lines.append("|-------|------|-------------|--------|")

        for name, metrics in all_metrics.items():
            if metrics.cv_mean_accuracy is not None:
                cv_acc = f"{metrics.cv_mean_accuracy:.1%} +/- {metrics.cv_std_accuracy:.1%}"
                cv_auc = f"{metrics.cv_mean_auc:.3f} +/- {metrics.cv_std_auc:.3f}" \
                    if metrics.cv_mean_auc else "N/A"
                lines.append(
                    f"| {metrics.model_name} | {metrics.mode} | {cv_acc} | {cv_auc} |"
                )

        lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
