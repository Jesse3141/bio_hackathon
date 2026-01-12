#!/usr/bin/env python3
"""
Compare current level distributions between control and modified cytosines.

This script analyzes the signal_at_cytosines.csv data to visualize and
quantify differences between canonical C and 5mC current levels.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats


def load_and_prepare_data(csv_file, bed_file):
    """Load signal data and merge with cytosine site metadata."""
    # Load signal data
    df = pd.read_csv(csv_file)

    # Load BED file for context scores
    bed = pd.read_csv(bed_file, sep='\t', header=None,
                      names=['chrom', 'start', 'end', 'mod_type', 'score', 'strand'])

    # Merge to get context scores
    df = df.merge(bed[['chrom', 'start', 'score']],
                  left_on=['chrom', 'position'],
                  right_on=['chrom', 'start'],
                  how='left')
    df = df.drop('start', axis=1)

    return df


def compute_statistics(df):
    """Compute comparison statistics between samples."""
    results = {}

    # Overall statistics
    for sample in df['sample'].unique():
        sample_data = df[df['sample'] == sample]['mean_current']
        results[f'{sample}_mean'] = sample_data.mean()
        results[f'{sample}_std'] = sample_data.std()
        results[f'{sample}_median'] = sample_data.median()
        results[f'{sample}_n'] = len(sample_data)

    # Statistical test (Welch's t-test)
    control = df[df['sample'] == 'control']['mean_current']
    modified = df[df['sample'] == 'modified']['mean_current']

    t_stat, p_value = stats.ttest_ind(control, modified, equal_var=False)
    results['t_statistic'] = t_stat
    results['p_value'] = p_value

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((control.std()**2 + modified.std()**2) / 2)
    cohens_d = (modified.mean() - control.mean()) / pooled_std
    results['cohens_d'] = cohens_d

    return results


def analyze_by_position(df):
    """Analyze signal differences at each cytosine position."""
    results = []

    for pos in sorted(df['position'].unique()):
        pos_data = df[df['position'] == pos]

        control = pos_data[pos_data['sample'] == 'control']['mean_current']
        modified = pos_data[pos_data['sample'] == 'modified']['mean_current']

        if len(control) > 5 and len(modified) > 5:
            t_stat, p_value = stats.ttest_ind(control, modified, equal_var=False)
            delta = modified.mean() - control.mean()
        else:
            t_stat, p_value, delta = np.nan, np.nan, np.nan

        results.append({
            'position': pos,
            'control_mean': control.mean() if len(control) > 0 else np.nan,
            'control_std': control.std() if len(control) > 0 else np.nan,
            'control_n': len(control),
            'modified_mean': modified.mean() if len(modified) > 0 else np.nan,
            'modified_std': modified.std() if len(modified) > 0 else np.nan,
            'modified_n': len(modified),
            'delta_mean': delta,
            't_statistic': t_stat,
            'p_value': p_value
        })

    return pd.DataFrame(results)


def analyze_by_context(df):
    """Analyze signal differences by sequence context (score from BED)."""
    results = []

    for score in sorted(df['score'].dropna().unique()):
        score_data = df[df['score'] == score]

        control = score_data[score_data['sample'] == 'control']['mean_current']
        modified = score_data[score_data['sample'] == 'modified']['mean_current']

        if len(control) > 5 and len(modified) > 5:
            t_stat, p_value = stats.ttest_ind(control, modified, equal_var=False)
            delta = modified.mean() - control.mean()
        else:
            t_stat, p_value, delta = np.nan, np.nan, np.nan

        results.append({
            'context_score': score,
            'control_mean': control.mean() if len(control) > 0 else np.nan,
            'control_n': len(control),
            'modified_mean': modified.mean() if len(modified) > 0 else np.nan,
            'modified_n': len(modified),
            'delta_mean': delta,
            'p_value': p_value
        })

    return pd.DataFrame(results)


def print_report(overall_stats, position_df, context_df):
    """Print analysis report to stdout."""
    print("=" * 60)
    print("SIGNAL COMPARISON REPORT: Control (C) vs Modified (5mC)")
    print("=" * 60)

    print("\n## Overall Statistics\n")
    print(f"Control:  mean = {overall_stats['control_mean']:.2f} pA, "
          f"std = {overall_stats['control_std']:.2f} pA, "
          f"n = {overall_stats['control_n']}")
    print(f"Modified: mean = {overall_stats['modified_mean']:.2f} pA, "
          f"std = {overall_stats['modified_std']:.2f} pA, "
          f"n = {overall_stats['modified_n']}")

    delta = overall_stats['modified_mean'] - overall_stats['control_mean']
    print(f"\nDelta (Modified - Control): {delta:.2f} pA ({delta/overall_stats['control_mean']*100:.1f}%)")
    print(f"Cohen's d effect size: {overall_stats['cohens_d']:.3f}")
    print(f"Welch's t-test: t = {overall_stats['t_statistic']:.2f}, "
          f"p = {overall_stats['p_value']:.2e}")

    print("\n## Analysis by Cytosine Position\n")
    print("Position | Control (pA) | Modified (pA) | Delta (pA) | p-value")
    print("-" * 65)
    for _, row in position_df.iterrows():
        print(f"{int(row['position']):8d} | "
              f"{row['control_mean']:12.2f} | "
              f"{row['modified_mean']:13.2f} | "
              f"{row['delta_mean']:10.2f} | "
              f"{row['p_value']:.2e}")

    if not context_df.empty:
        print("\n## Analysis by Sequence Context (BED Score)\n")
        print("Context | Control (pA) | Modified (pA) | Delta (pA) | p-value")
        print("-" * 65)
        for _, row in context_df.iterrows():
            print(f"{int(row['context_score']):7d} | "
                  f"{row['control_mean']:12.2f} | "
                  f"{row['modified_mean']:13.2f} | "
                  f"{row['delta_mean']:10.2f} | "
                  f"{row['p_value']:.2e}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if overall_stats['p_value'] < 0.001 and abs(overall_stats['cohens_d']) > 0.2:
        print(f"\nStatistically significant difference detected (p < 0.001)")
        print(f"Effect size: {'large' if abs(overall_stats['cohens_d']) > 0.8 else 'medium' if abs(overall_stats['cohens_d']) > 0.5 else 'small'}")
        print(f"\n5mC cytosines produce {'higher' if delta > 0 else 'lower'} current levels than canonical C")
        print(f"This difference ({delta:.1f} pA) can be used for modification detection.")
    else:
        print("\nNo significant difference detected or effect size too small.")

    print()


def save_for_hmm_training(df, output_prefix):
    """Save data in format suitable for HMM training."""

    # Aggregate per read at each position for HMM training
    hmm_data = df.groupby(['sample', 'chrom', 'position', 'read_id']).agg({
        'mean_current': 'first',
        'std_current': 'first',
        'dwell_time': 'first',
        'n_samples': 'first'
    }).reset_index()

    # Create separate files for control and modified
    control_data = hmm_data[hmm_data['sample'] == 'control']
    modified_data = hmm_data[hmm_data['sample'] == 'modified']

    control_data.to_csv(f'{output_prefix}_control.csv', index=False)
    modified_data.to_csv(f'{output_prefix}_modified.csv', index=False)

    print(f"\nSaved HMM training data:", file=sys.stderr)
    print(f"  {output_prefix}_control.csv  ({len(control_data)} records)", file=sys.stderr)
    print(f"  {output_prefix}_modified.csv ({len(modified_data)} records)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Compare control vs modified signal distributions')
    parser.add_argument('--input', required=True, help='Signal CSV from extract_signal_at_positions.py')
    parser.add_argument('--bed', required=True, help='BED file with cytosine positions')
    parser.add_argument('--hmm-output', default='output/hmm_training_data',
                        help='Prefix for HMM training data output')

    args = parser.parse_args()

    # Load data
    print("Loading data...", file=sys.stderr)
    df = load_and_prepare_data(args.input, args.bed)

    # Compute statistics
    print("Computing statistics...", file=sys.stderr)
    overall_stats = compute_statistics(df)
    position_df = analyze_by_position(df)
    context_df = analyze_by_context(df)

    # Print report
    print_report(overall_stats, position_df, context_df)

    # Save for HMM training
    save_for_hmm_training(df, args.hmm_output)


if __name__ == '__main__':
    main()
