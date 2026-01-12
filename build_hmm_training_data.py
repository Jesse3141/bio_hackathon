#!/usr/bin/env python3
"""
Build HMM training data from extracted signal measurements.

This script creates training data for the circuit board HMM model by:
1. Computing emission distributions (Gaussian parameters) for each cytosine state
2. Extracting per-read signal sequences for model training
3. Generating k-mer specific current models from the data
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


def compute_emission_parameters(df, sample_name):
    """Compute Gaussian emission parameters for HMM states.

    Returns dict mapping position -> {mean, std, n} for the sample.
    """
    sample_df = df[df['sample'] == sample_name]
    params = {}

    for pos in sorted(sample_df['position'].unique()):
        pos_data = sample_df[sample_df['position'] == pos]['mean_current']
        params[int(pos)] = {
            'mean': float(pos_data.mean()),
            'std': float(pos_data.std()),
            'n': int(len(pos_data))
        }

    return params


def fit_mixture_model(control_df, modified_df):
    """Fit a simple mixture model for C vs 5mC classification.

    Returns parameters for binary classification at each position.
    """
    results = []

    positions = sorted(set(control_df['position'].unique()) &
                       set(modified_df['position'].unique()))

    for pos in positions:
        c_data = control_df[control_df['position'] == pos]['mean_current']
        m_data = modified_df[modified_df['position'] == pos]['mean_current']

        # Compute threshold using Fisher's linear discriminant
        c_mean, c_std = c_data.mean(), c_data.std()
        m_mean, m_std = m_data.mean(), m_data.std()

        # Optimal threshold (equal error rate point)
        pooled_std = np.sqrt((c_std**2 + m_std**2) / 2)
        threshold = (c_mean + m_mean) / 2

        # Compute classification accuracy at threshold
        c_correct = (c_data < threshold).sum() / len(c_data)
        m_correct = (m_data >= threshold).sum() / len(m_data)
        accuracy = (c_correct * len(c_data) + m_correct * len(m_data)) / (len(c_data) + len(m_data))

        # Likelihood ratio test statistics
        d_prime = (m_mean - c_mean) / pooled_std

        results.append({
            'position': pos,
            'c_mean': c_mean,
            'c_std': c_std,
            'm_mean': m_mean,
            'm_std': m_std,
            'threshold': threshold,
            'd_prime': d_prime,
            'accuracy': accuracy
        })

    return pd.DataFrame(results)


def create_kmer_model(df, bed_file, reference_fasta):
    """Create k-mer specific current model from signal data.

    Combines signal measurements with sequence context to build
    a k-mer -> current distribution lookup table.
    """
    from Bio import SeqIO

    # Load reference sequences
    ref_seqs = {}
    for record in SeqIO.parse(reference_fasta, 'fasta'):
        ref_seqs[record.id] = str(record.seq)

    # Load BED for context info
    bed = pd.read_csv(bed_file, sep='\t', header=None,
                      names=['chrom', 'start', 'end', 'mod_type', 'score', 'strand'])

    # Extract 5-mer context for each position
    kmer_data = []

    for _, row in df.iterrows():
        chrom = row['chrom']
        pos = int(row['position'])

        if chrom not in ref_seqs:
            continue

        seq = ref_seqs[chrom]

        # Extract 5-mer centered on position (2 bases either side)
        if pos >= 2 and pos < len(seq) - 2:
            kmer = seq[pos-2:pos+3]
        else:
            continue

        kmer_data.append({
            'kmer': kmer,
            'sample': row['sample'],
            'mean_current': row['mean_current'],
            'std_current': row['std_current'],
            'dwell_time': row['dwell_time']
        })

    kmer_df = pd.DataFrame(kmer_data)

    # Aggregate by k-mer and sample
    kmer_model = kmer_df.groupby(['kmer', 'sample']).agg({
        'mean_current': ['mean', 'std', 'count'],
        'std_current': 'mean',
        'dwell_time': ['mean', 'std']
    }).reset_index()

    kmer_model.columns = ['kmer', 'sample', 'current_mean', 'current_std', 'n',
                          'noise_mean', 'dwell_mean', 'dwell_std']

    return kmer_model


def export_for_pomegranate(emission_params, output_file):
    """Export emission parameters in format for pomegranate HMM."""
    pomegranate_states = []

    for sample_name, positions in emission_params.items():
        for pos, params in positions.items():
            state_name = f"{sample_name}_pos{pos}"
            pomegranate_states.append({
                'name': state_name,
                'distribution': 'Normal',
                'parameters': {
                    'mean': params['mean'],
                    'std': params['std']
                },
                'n_observations': params['n']
            })

    with open(output_file, 'w') as f:
        json.dump({'states': pomegranate_states}, f, indent=2)


def export_for_circuit_board_hmm(control_params, modified_params, output_file):
    """Export in format compatible with epigenetics.py circuit board HMM.

    The circuit board HMM expects emission distributions per position,
    with separate distributions for C, mC, and hmC states.
    """
    # Build distribution list: for each cytosine position, we have
    # C and mC emission parameters
    distributions = []

    positions = sorted(set(control_params.keys()) & set(modified_params.keys()))

    for pos in positions:
        c_params = control_params[pos]
        m_params = modified_params[pos]

        distributions.append({
            'position': pos,
            'C': {
                'type': 'NormalDistribution',
                'mean': c_params['mean'],
                'std': c_params['std'],
                'n': c_params['n']
            },
            'mC': {
                'type': 'NormalDistribution',
                'mean': m_params['mean'],
                'std': m_params['std'],
                'n': m_params['n']
            }
        })

    output = {
        'model_type': 'circuit_board_hmm',
        'n_positions': len(positions),
        'positions': positions,
        'distributions': distributions,
        'metadata': {
            'source': 'signal_alignment_pipeline',
            'control_sample': 'canonical_C',
            'modified_sample': '5mC'
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)


def export_sequences_for_training(df, output_file):
    """Export per-read signal sequences for HMM training.

    Format: One read per line, tab-separated current values at each position.
    """
    # Group by read and pivot to get positions as columns
    pivoted = df.pivot_table(
        index=['sample', 'chrom', 'read_id'],
        columns='position',
        values='mean_current',
        aggfunc='first'
    ).reset_index()

    pivoted.to_csv(output_file, index=False)
    return pivoted


def main():
    parser = argparse.ArgumentParser(description='Build HMM training data')
    parser.add_argument('--input', required=True, help='Signal CSV from extract_signal_at_positions.py')
    parser.add_argument('--bed', required=True, help='BED file with cytosine positions')
    parser.add_argument('--reference', required=True, help='Reference FASTA file')
    parser.add_argument('--output-dir', default='output', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...", file=sys.stderr)
    df = pd.read_csv(args.input)

    # Compute emission parameters
    print("Computing emission parameters...", file=sys.stderr)
    control_params = compute_emission_parameters(df, 'control')
    modified_params = compute_emission_parameters(df, 'modified')

    # Export in various formats
    print("Exporting training data...", file=sys.stderr)

    # 1. Pomegranate format
    export_for_pomegranate(
        {'control': control_params, 'modified': modified_params},
        output_dir / 'hmm_emission_params_pomegranate.json'
    )

    # 2. Circuit board HMM format
    export_for_circuit_board_hmm(
        control_params, modified_params,
        output_dir / 'hmm_emission_params_circuit_board.json'
    )

    # 3. Per-read sequences
    pivoted = export_sequences_for_training(df, output_dir / 'hmm_training_sequences.csv')

    # 4. Classification model parameters
    print("Fitting classification model...", file=sys.stderr)
    control_df = df[df['sample'] == 'control']
    modified_df = df[df['sample'] == 'modified']
    class_params = fit_mixture_model(control_df, modified_df)
    class_params.to_csv(output_dir / 'classification_parameters.csv', index=False)

    # 5. K-mer model (if biopython available)
    try:
        print("Building k-mer model...", file=sys.stderr)
        kmer_model = create_kmer_model(df, args.bed, args.reference)
        kmer_model.to_csv(output_dir / 'kmer_current_model.csv', index=False)
    except ImportError:
        print("  Skipping k-mer model (biopython not installed)", file=sys.stderr)

    # Print summary
    print("\n=== HMM Training Data Summary ===", file=sys.stderr)
    print(f"Cytosine positions: {len(control_params)}", file=sys.stderr)
    print(f"Control observations: {sum(p['n'] for p in control_params.values())}", file=sys.stderr)
    print(f"Modified observations: {sum(p['n'] for p in modified_params.values())}", file=sys.stderr)
    print(f"Unique reads: {df['read_id'].nunique()}", file=sys.stderr)

    print(f"\nFiles created in {output_dir}/:", file=sys.stderr)
    print("  - hmm_emission_params_pomegranate.json", file=sys.stderr)
    print("  - hmm_emission_params_circuit_board.json", file=sys.stderr)
    print("  - hmm_training_sequences.csv", file=sys.stderr)
    print("  - classification_parameters.csv", file=sys.stderr)
    print("  - kmer_current_model.csv (if biopython installed)", file=sys.stderr)

    # Print classification parameters
    print("\n=== Classification Model ===", file=sys.stderr)
    print(class_params.to_string(index=False), file=sys.stderr)

    # Print overall classification accuracy
    mean_accuracy = class_params['accuracy'].mean()
    mean_dprime = class_params['d_prime'].mean()
    print(f"\nMean classification accuracy: {mean_accuracy:.1%}", file=sys.stderr)
    print(f"Mean d': {mean_dprime:.2f}", file=sys.stderr)


if __name__ == '__main__':
    main()
