#!/usr/bin/env python3
"""
Build HMM training data for 3-way cytosine modification classification.

Generates Gaussian emission parameters and training sequences for:
- Canonical C (unmodified)
- 5mC (5-methylcytosine)
- 5hmC (5-hydroxymethylcytosine)
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


def load_signals(csv_file):
    """Load signal measurements from CSV."""
    df = pd.read_csv(csv_file)
    return df


def compute_statistics(df, sample_name):
    """Compute per-position statistics for a sample."""
    sample_df = df[df['sample'] == sample_name]

    stats_list = []
    for position in sorted(sample_df['position'].unique()):
        pos_data = sample_df[sample_df['position'] == position]['mean_current']

        stats_list.append({
            'position': int(position),
            'mean': float(pos_data.mean()),
            'std': float(pos_data.std()),
            'count': int(len(pos_data)),
            'min': float(pos_data.min()),
            'max': float(pos_data.max()),
            'median': float(pos_data.median())
        })

    return stats_list


def compute_pairwise_statistics(df, sample1, sample2):
    """Compute statistical differences between two samples."""
    results = []

    df1 = df[df['sample'] == sample1]
    df2 = df[df['sample'] == sample2]

    positions = sorted(set(df1['position'].unique()) & set(df2['position'].unique()))

    for pos in positions:
        data1 = df1[df1['position'] == pos]['mean_current']
        data2 = df2[df2['position'] == pos]['mean_current']

        if len(data1) > 1 and len(data2) > 1:
            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)

            # Cohen's d effect size
            pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
            cohens_d = (data2.mean() - data1.mean()) / pooled_std if pooled_std > 0 else 0

            results.append({
                'position': int(pos),
                f'{sample1}_mean': float(data1.mean()),
                f'{sample1}_std': float(data1.std()),
                f'{sample1}_n': int(len(data1)),
                f'{sample2}_mean': float(data2.mean()),
                f'{sample2}_std': float(data2.std()),
                f'{sample2}_n': int(len(data2)),
                'delta': float(data2.mean() - data1.mean()),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d)
            })

    return results


def build_circuit_board_params(df, output_file):
    """Build emission parameters in circuit board HMM format."""
    samples = ['control', '5mC', '5hmC']
    sample_labels = {'control': 'C', '5mC': 'mC', '5hmC': 'hmC'}

    params = {
        'description': '3-way cytosine modification HMM emission parameters',
        'samples': samples,
        'distributions': []
    }

    positions = sorted(df['position'].unique())

    for pos in positions:
        dist_entry = {'position': int(pos)}

        for sample in samples:
            sample_data = df[(df['sample'] == sample) & (df['position'] == pos)]['mean_current']

            if len(sample_data) > 0:
                label = sample_labels[sample]
                dist_entry[label] = {
                    'mean': float(sample_data.mean()),
                    'std': float(sample_data.std()),
                    'count': int(len(sample_data))
                }

        params['distributions'].append(dist_entry)

    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Saved circuit board params to {output_file}", file=sys.stderr)


def build_pomegranate_params(df, output_file):
    """Build emission parameters in pomegranate-compatible format."""
    samples = ['control', '5mC', '5hmC']
    sample_labels = {'control': 'C', '5mC': 'mC', '5hmC': 'hmC'}

    params = {
        'type': 'pomegranate_hmm_emissions',
        'modification_states': ['C', 'mC', 'hmC'],
        'positions': []
    }

    positions = sorted(df['position'].unique())

    for pos in positions:
        pos_entry = {'position': int(pos), 'states': {}}

        for sample in samples:
            sample_data = df[(df['sample'] == sample) & (df['position'] == pos)]['mean_current']

            if len(sample_data) > 0:
                label = sample_labels[sample]
                pos_entry['states'][label] = {
                    'distribution': 'NormalDistribution',
                    'parameters': {
                        'mean': float(sample_data.mean()),
                        'std': float(sample_data.std())
                    }
                }

        params['positions'].append(pos_entry)

    with open(output_file, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Saved pomegranate params to {output_file}", file=sys.stderr)


def build_training_sequences(df, output_file):
    """Build per-read training sequences for HMM training."""
    # Get reads that have measurements at multiple positions
    read_counts = df.groupby(['sample', 'read_id']).size()

    sequences = []

    for (sample, read_id), count in read_counts.items():
        if count >= 3:  # At least 3 positions
            read_data = df[(df['sample'] == sample) & (df['read_id'] == read_id)]
            read_data = read_data.sort_values('position')

            seq_entry = {
                'sample': sample,
                'read_id': read_id,
                'positions': read_data['position'].tolist(),
                'currents': read_data['mean_current'].tolist(),
                'n_positions': len(read_data)
            }
            sequences.append(seq_entry)

    seq_df = pd.DataFrame(sequences)
    seq_df.to_csv(output_file, index=False)

    print(f"Saved {len(seq_df)} training sequences to {output_file}", file=sys.stderr)

    # Print summary
    for sample in ['control', '5mC', '5hmC']:
        sample_seqs = seq_df[seq_df['sample'] == sample]
        if len(sample_seqs) > 0:
            print(f"  {sample}: {len(sample_seqs)} sequences", file=sys.stderr)


def print_summary_report(df, output_dir):
    """Print and save a summary report of the 3-way comparison."""
    report_lines = []

    report_lines.append("=" * 60)
    report_lines.append("3-WAY CYTOSINE MODIFICATION SIGNAL ANALYSIS")
    report_lines.append("=" * 60)
    report_lines.append("")

    # Overall statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 40)

    for sample in ['control', '5mC', '5hmC']:
        sample_data = df[df['sample'] == sample]['mean_current']
        if len(sample_data) > 0:
            report_lines.append(f"{sample:10s}: mean={sample_data.mean():7.2f} pA, "
                              f"std={sample_data.std():6.2f} pA, n={len(sample_data)}")

    report_lines.append("")

    # Pairwise comparisons
    report_lines.append("PAIRWISE COMPARISONS (mean delta)")
    report_lines.append("-" * 40)

    comparisons = [
        ('control', '5mC', 'C vs 5mC'),
        ('control', '5hmC', 'C vs 5hmC'),
        ('5mC', '5hmC', '5mC vs 5hmC')
    ]

    for sample1, sample2, label in comparisons:
        data1 = df[df['sample'] == sample1]['mean_current']
        data2 = df[df['sample'] == sample2]['mean_current']

        if len(data1) > 0 and len(data2) > 0:
            delta = data2.mean() - data1.mean()
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
            cohens_d = delta / pooled_std if pooled_std > 0 else 0

            report_lines.append(f"{label:15s}: delta={delta:+7.2f} pA, "
                              f"Cohen's d={cohens_d:.3f}, p={p_value:.2e}")

    report_lines.append("")

    # Per-position summary
    report_lines.append("PER-POSITION MEAN CURRENTS (pA)")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Position':>10s} {'Control':>10s} {'5mC':>10s} {'5hmC':>10s} {'C-mC':>8s} {'C-hmC':>8s}")

    positions = sorted(df['position'].unique())
    for pos in positions:
        values = {}
        for sample in ['control', '5mC', '5hmC']:
            sample_data = df[(df['sample'] == sample) & (df['position'] == pos)]['mean_current']
            values[sample] = sample_data.mean() if len(sample_data) > 0 else float('nan')

        delta_mc = values['5mC'] - values['control']
        delta_hmc = values['5hmC'] - values['control']

        report_lines.append(f"{pos:>10d} {values['control']:>10.2f} {values['5mC']:>10.2f} "
                          f"{values['5hmC']:>10.2f} {delta_mc:>+8.2f} {delta_hmc:>+8.2f}")

    report_lines.append("")
    report_lines.append("=" * 60)

    # Print to stderr
    for line in report_lines:
        print(line, file=sys.stderr)

    # Save to file
    report_file = output_dir / '3way_analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nSaved report to {report_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Build HMM training data for 3-way classification')
    parser.add_argument('--input', required=True, help='Input CSV with signal measurements')
    parser.add_argument('--output-dir', required=True, help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading signal data...", file=sys.stderr)
    df = load_signals(args.input)
    print(f"  Loaded {len(df)} measurements", file=sys.stderr)

    # Check we have all 3 samples
    samples = df['sample'].unique()
    print(f"  Samples found: {list(samples)}", file=sys.stderr)

    expected = {'control', '5mC', '5hmC'}
    if not expected.issubset(set(samples)):
        print(f"Warning: Expected samples {expected}, found {set(samples)}", file=sys.stderr)

    # Build output files
    print("\nBuilding HMM emission parameters...", file=sys.stderr)

    build_circuit_board_params(df, output_dir / 'hmm_3way_circuit_board.json')
    build_pomegranate_params(df, output_dir / 'hmm_3way_pomegranate.json')
    build_training_sequences(df, output_dir / 'hmm_3way_training_sequences.csv')

    # Save pairwise statistics
    print("\nComputing pairwise statistics...", file=sys.stderr)

    for sample1, sample2 in [('control', '5mC'), ('control', '5hmC'), ('5mC', '5hmC')]:
        stats_data = compute_pairwise_statistics(df, sample1, sample2)
        stats_df = pd.DataFrame(stats_data)
        stats_file = output_dir / f'pairwise_stats_{sample1}_vs_{sample2}.csv'
        stats_df.to_csv(stats_file, index=False)
        print(f"  Saved {stats_file}", file=sys.stderr)

    # Print summary report
    print("\n", file=sys.stderr)
    print_summary_report(df, output_dir)


if __name__ == '__main__':
    main()
