#!/usr/bin/env python
"""
Run all Full-Sequence HMM evaluation configurations.

Evaluates 4 configurations as specified in EVALUATION_PLAN.md:
1. binary_single  - Binary mode (C/5mC), single adapter emissions
2. binary_pooled  - Binary mode (C/5mC), pooled emissions (all adapters)
3. 3way_single    - 3-way mode (C/5mC/5hmC), single adapter emissions
4. 3way_pooled    - 3-way mode (C/5mC/5hmC), pooled emissions

Generates:
- Per-configuration results (metrics.json, plots)
- Comparison table (comparison_table.md)
- Summary CSV for analysis

Usage:
    python run_all_evaluations.py --output results/full_evaluation/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from methylation_hmm.full_sequence_hmm import FullSequenceHMM
from methylation_hmm.emission_params import (
    compute_emission_params_from_full_csv,
    FullSequenceEmissionParams,
)
from methylation_hmm.evaluation import (
    UnifiedMetrics,
    save_json,
    save_csv,
    generate_plots,
)
from methylation_hmm.evaluation.output_formatters import generate_comparison_table
from methylation_hmm.evaluation.site_type_metrics import (
    load_bed_with_site_types,
    get_site_type_positions,
)


# Configuration definitions
CONFIGURATIONS = [
    {"name": "binary_single", "mode": "binary", "emission_source": "single"},
    {"name": "binary_pooled", "mode": "binary", "emission_source": "pooled"},
    {"name": "3way_single", "mode": "3way", "emission_source": "single"},
    {"name": "3way_pooled", "mode": "3way", "emission_source": "pooled"},
]

SINGLE_ADAPTER = "5mers_rand_ref_adapter_01"


def get_default_paths() -> Dict[str, Path]:
    """Get default file paths."""
    return {
        "full_signal_csv": PROJECT_ROOT / "output" / "rep1" / "signal_full_sequence.csv",
        "bed_file": PROJECT_ROOT / "nanopore_ref_data" / "all_5mers_C_sites.bed",
        "output_dir": PROJECT_ROOT / "results" / "full_evaluation",
    }


def run_single_evaluation(
    config: Dict,
    signal_csv: Path,
    output_dir: Path,
    bed_file: Path,
    test_split: float = 0.2,
    cv_folds: int = 5,
    seed: int = 42,
    no_plots: bool = False,
) -> UnifiedMetrics:
    """
    Run evaluation for a single configuration.

    Args:
        config: Configuration dict with name, mode, emission_source
        signal_csv: Path to signal_full_sequence.csv
        output_dir: Base output directory
        bed_file: Path to BED file with site types
        test_split: Test set fraction
        cv_folds: Number of CV folds
        seed: Random seed
        no_plots: Skip plot generation

    Returns:
        UnifiedMetrics for this configuration
    """
    from methylation_hmm.cli.run_evaluation import (
        load_full_sequence_data,
        evaluate_full_sequence_hmm,
    )

    name = config["name"]
    mode = config["mode"]
    emission_source = config["emission_source"]

    print(f"\n{'=' * 70}")
    print(f"CONFIGURATION: {name}")
    print(f"{'=' * 70}")
    print(f"Mode: {mode}")
    print(f"Emission source: {emission_source}")

    # Determine adapter
    adapter = SINGLE_ADAPTER if emission_source == "single" else None

    # Create output directory for this config
    config_output = output_dir / name
    config_output.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    full_df, train_df, test_df = load_full_sequence_data(
        csv_path=signal_csv,
        adapter=adapter,
        mode=mode,
        test_split=test_split,
        seed=seed,
    )

    # Compute emission parameters
    print("Computing emission parameters...")
    emission_params = compute_emission_params_from_full_csv(
        str(signal_csv),
        adapter=adapter,
        mode=mode,
    )

    # Save emission parameters
    params_path = config_output / "emission_params.json"
    emission_params.save(str(params_path))

    # Load site type positions
    print("Loading site type annotations...")
    bed_df = load_bed_with_site_types(str(bed_file))
    # Use representative adapter for site types (consistent across adapters at same positions)
    site_type_positions = get_site_type_positions(bed_df, SINGLE_ADAPTER)

    # Evaluate
    print("Evaluating...")
    metrics = evaluate_full_sequence_hmm(
        test_df=test_df,
        emission_params=emission_params,
        mode=mode,
        cv_folds=cv_folds,
        train_df=train_df,
        site_type_positions=site_type_positions,
    )

    # Print summary
    print(f"\nResults for {name}:")
    print(f"  Overall accuracy: {metrics.overall_accuracy:.1%}")
    print(f"  Top 25% accuracy: {metrics.accuracy_at_top_25pct:.1%}")
    if metrics.cv_mean_accuracy:
        print(f"  CV accuracy: {metrics.cv_mean_accuracy:.1%} +/- {metrics.cv_std_accuracy:.1%}")

    # Save results
    metrics_dict = metrics.to_dict()
    metrics_dict['emission_params_summary'] = emission_params.summary()
    metrics_dict['configuration'] = config

    json_path = config_output / "metrics.json"
    with open(json_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            return obj
        json.dump(metrics_dict, f, indent=2, default=convert)

    save_csv(metrics, config_output / "metrics.csv")

    if not no_plots:
        generate_plots(metrics, config_output)

    return metrics


def generate_summary_table(
    all_metrics: Dict[str, UnifiedMetrics],
    output_path: Path,
) -> pd.DataFrame:
    """Generate a summary DataFrame and save as CSV."""
    rows = []

    for name, metrics in all_metrics.items():
        row = {
            'configuration': name,
            'mode': metrics.mode,
            'emission_source': 'single' if 'single' in name else 'pooled',
            'overall_accuracy': metrics.overall_accuracy,
            'top_25pct_accuracy': metrics.accuracy_at_top_25pct,
            'top_50pct_accuracy': metrics.accuracy_at_top_50pct,
            'auc_macro': metrics.auc_macro if metrics.auc_macro else 0.0,
            'cv_mean_accuracy': metrics.cv_mean_accuracy if metrics.cv_mean_accuracy else 0.0,
            'cv_std_accuracy': metrics.cv_std_accuracy if metrics.cv_std_accuracy else 0.0,
            'n_samples': metrics.n_samples,
        }

        # Per-class accuracy
        for class_name, acc in metrics.per_class_accuracy.items():
            row[f'{class_name}_accuracy'] = acc

        # Site-type accuracy
        for st_name in ['non_cpg', 'cpg', 'homopolymer']:
            row[f'{st_name}_accuracy'] = metrics.accuracy_by_site_type.get(st_name, 0.0)
            row[f'{st_name}_n_samples'] = metrics.n_samples_by_site_type.get(st_name, 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df


def generate_markdown_report(
    all_metrics: Dict[str, UnifiedMetrics],
    output_path: Path,
) -> None:
    """Generate comprehensive markdown comparison report."""
    lines = []
    lines.append("# Full-Sequence HMM Evaluation Results")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("This report compares Full-Sequence HMM classifiers that model the entire 155bp")
    lines.append("reference sequence. The key variable is **emission parameter source**:")
    lines.append("- **Single adapter**: Context-specific parameters (lower variance)")
    lines.append("- **Pooled**: Parameters from all adapters (higher variance)")
    lines.append("")

    # Summary table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Configuration | Mode | Emission | Overall | Top 25% | Top 50% | AUC | CV Acc |")
    lines.append("|---------------|------|----------|---------|---------|---------|-----|--------|")

    for name, metrics in all_metrics.items():
        mode = metrics.mode
        source = 'single' if 'single' in name else 'pooled'
        overall = f"{metrics.overall_accuracy:.1%}"
        top25 = f"{metrics.accuracy_at_top_25pct:.1%}"
        top50 = f"{metrics.accuracy_at_top_50pct:.1%}"
        auc = f"{metrics.auc_macro:.3f}" if metrics.auc_macro else "N/A"
        cv = f"{metrics.cv_mean_accuracy:.1%}" if metrics.cv_mean_accuracy else "N/A"

        lines.append(f"| {name} | {mode} | {source} | {overall} | {top25} | {top50} | {auc} | {cv} |")

    lines.append("")

    # Best configuration
    best_config = max(all_metrics.items(), key=lambda x: x[1].overall_accuracy)
    lines.append("## Best Configuration")
    lines.append("")
    lines.append(f"**{best_config[0]}** achieved the highest overall accuracy: {best_config[1].overall_accuracy:.1%}")
    lines.append("")

    # Per-class accuracy comparison
    lines.append("## Per-Class Accuracy")
    lines.append("")
    lines.append("| Configuration | C | 5mC | 5hmC |")
    lines.append("|---------------|---|-----|------|")

    for name, metrics in all_metrics.items():
        c_acc = f"{metrics.per_class_accuracy.get('C', 0):.1%}"
        mc_acc = f"{metrics.per_class_accuracy.get('5mC', 0):.1%}"
        hmc_acc = f"{metrics.per_class_accuracy.get('5hmC', 0):.1%}" if '5hmC' in metrics.per_class_accuracy else "N/A"
        lines.append(f"| {name} | {c_acc} | {mc_acc} | {hmc_acc} |")

    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Compare single vs pooled for binary
    binary_single = all_metrics.get('binary_single')
    binary_pooled = all_metrics.get('binary_pooled')

    if binary_single and binary_pooled:
        diff = binary_single.overall_accuracy - binary_pooled.overall_accuracy
        better = "single" if diff > 0 else "pooled"
        lines.append(f"### Binary Mode")
        lines.append(f"- Single adapter: {binary_single.overall_accuracy:.1%}")
        lines.append(f"- Pooled: {binary_pooled.overall_accuracy:.1%}")
        lines.append(f"- **Difference: {abs(diff):.1%} ({better} is better)**")
        lines.append("")

    # Compare single vs pooled for 3-way
    threeway_single = all_metrics.get('3way_single')
    threeway_pooled = all_metrics.get('3way_pooled')

    if threeway_single and threeway_pooled:
        diff = threeway_single.overall_accuracy - threeway_pooled.overall_accuracy
        better = "single" if diff > 0 else "pooled"
        lines.append(f"### 3-Way Mode")
        lines.append(f"- Single adapter: {threeway_single.overall_accuracy:.1%}")
        lines.append(f"- Pooled: {threeway_pooled.overall_accuracy:.1%}")
        lines.append(f"- **Difference: {abs(diff):.1%} ({better} is better)**")
        lines.append("")

    # Confidence analysis
    lines.append("## Confidence-Stratified Analysis")
    lines.append("")
    lines.append("| Configuration | Top 25% | Top 50% | All |")
    lines.append("|---------------|---------|---------|-----|")

    for name, metrics in all_metrics.items():
        top25 = f"{metrics.accuracy_at_top_25pct:.1%}"
        top50 = f"{metrics.accuracy_at_top_50pct:.1%}"
        all_acc = f"{metrics.accuracy_at_100pct:.1%}"
        lines.append(f"| {name} | {top25} | {top50} | {all_acc} |")

    lines.append("")
    lines.append("This shows accuracy when only considering the most confident predictions.")
    lines.append("")

    # Site-type accuracy
    lines.append("## Accuracy by Site Type")
    lines.append("")
    lines.append("Per-position accuracy grouped by cytosine context:")
    lines.append("- **non_cpg**: C not followed by G")
    lines.append("- **cpg**: CpG dinucleotide")
    lines.append("- **homopolymer**: CC run (adjacent cytosines)")
    lines.append("")
    lines.append("| Configuration | non_cpg | cpg | homopolymer |")
    lines.append("|---------------|---------|-----|-------------|")

    for name, metrics in all_metrics.items():
        non_cpg = f"{metrics.accuracy_by_site_type.get('non_cpg', 0):.1%}"
        cpg = f"{metrics.accuracy_by_site_type.get('cpg', 0):.1%}"
        homo = f"{metrics.accuracy_by_site_type.get('homopolymer', 0):.1%}"
        lines.append(f"| {name} | {non_cpg} | {cpg} | {homo} |")

    lines.append("")

    # Cross-validation
    lines.append("## Cross-Validation Results")
    lines.append("")

    has_cv = any(m.cv_mean_accuracy for m in all_metrics.values())
    if has_cv:
        lines.append("| Configuration | Mean Acc | Std |")
        lines.append("|---------------|----------|-----|")
        for name, metrics in all_metrics.items():
            if metrics.cv_mean_accuracy:
                lines.append(f"| {name} | {metrics.cv_mean_accuracy:.1%} | {metrics.cv_std_accuracy:.1%} |")
    else:
        lines.append("Cross-validation was not run.")

    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Run all Full-Sequence HMM evaluation configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

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
        help="Path to BED file with site type annotations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults["output_dir"],
        help="Output directory for all results",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction for test set (default: 0.2)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds (0 to skip)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[c["name"] for c in CONFIGURATIONS],
        help="Specific configurations to run (default: all)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL-SEQUENCE HMM EVALUATION SUITE")
    print("=" * 70)
    print(f"Signal CSV: {args.signal_csv}")
    print(f"Output: {args.output}")
    print(f"Test split: {args.test_split}")
    print(f"CV folds: {args.cv_folds}")
    print()

    # Determine which configurations to run
    if args.configs:
        configs_to_run = [c for c in CONFIGURATIONS if c["name"] in args.configs]
    else:
        configs_to_run = CONFIGURATIONS

    print(f"Running {len(configs_to_run)} configurations:")
    for c in configs_to_run:
        print(f"  - {c['name']} ({c['mode']}, {c['emission_source']})")

    # Run evaluations
    all_metrics = {}

    for config in configs_to_run:
        try:
            metrics = run_single_evaluation(
                config=config,
                signal_csv=args.signal_csv,
                output_dir=args.output,
                bed_file=args.bed_file,
                test_split=args.test_split,
                cv_folds=args.cv_folds,
                seed=args.seed,
                no_plots=args.no_plots,
            )
            all_metrics[config["name"]] = metrics
        except Exception as e:
            print(f"\nERROR running {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison outputs
    if len(all_metrics) > 0:
        print("\n" + "=" * 70)
        print("GENERATING COMPARISON REPORTS")
        print("=" * 70)

        # Summary CSV
        summary_csv = args.output / "summary.csv"
        summary_df = generate_summary_table(all_metrics, summary_csv)
        print(f"Saved: {summary_csv}")

        # Markdown report
        report_md = args.output / "comparison_table.md"
        generate_markdown_report(all_metrics, report_md)
        print(f"Saved: {report_md}")

        # Print final summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print()
        print(summary_df[['configuration', 'overall_accuracy', 'top_25pct_accuracy', 'cv_mean_accuracy']].to_string(index=False))

        # Best configuration
        best = max(all_metrics.items(), key=lambda x: x[1].overall_accuracy)
        print(f"\nBest configuration: {best[0]} ({best[1].overall_accuracy:.1%})")

    print("\nDone!")
    print(f"All results saved to: {args.output}")


if __name__ == "__main__":
    main()
