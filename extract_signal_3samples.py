#!/usr/bin/env python3
"""
Extract per-position current levels from signal-aligned BAM files for 3 samples.

This script reads signal-aligned BAMs produced by uncalled4 and extracts
current level statistics at cytosine positions for control, 5mC, and 5hmC samples.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pysam
import pod5


def parse_move_table(read):
    """Parse the move table (mv tag) from a BAM read."""
    mv_tag = read.get_tag('mv') if read.has_tag('mv') else None

    if mv_tag is None:
        return None, None

    stride = mv_tag[0]
    moves = np.array(mv_tag[1:], dtype=np.int8)

    return stride, moves


def get_signal_boundaries(moves, stride, signal_start=0):
    """Convert move table to signal sample boundaries for each base."""
    boundaries = []
    current_sample = signal_start

    for i, move in enumerate(moves):
        if move == 1:
            start = current_sample
            j = i + 1
            while j < len(moves) and moves[j] == 0:
                j += 1
            end = signal_start + j * stride
            boundaries.append((start, end))
        current_sample += stride

    return boundaries


def extract_signal_at_position(signal, boundaries, base_position):
    """Extract signal values for a specific base position."""
    if base_position < 0 or base_position >= len(boundaries):
        return None

    start, end = boundaries[base_position]
    if start >= len(signal) or end > len(signal):
        return None

    return signal[start:end]


def load_cytosine_sites(bed_file):
    """Load cytosine positions from BED file."""
    df = pd.read_csv(bed_file, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'name', 'score', 'strand'])
    return df


def process_bam_with_pod5(bam_file, pod5_file, bed_df, sample_name):
    """Extract signal at cytosine positions from signal-aligned BAM."""
    results = []

    sites_by_ref = bed_df.groupby('chrom')['start'].apply(set).to_dict()

    # Load POD5 for signal data
    print(f"  Loading POD5 signals from {pod5_file}...", file=sys.stderr)
    pod5_reader = pod5.Reader(pod5_file)
    read_signals = {}
    for read_record in pod5_reader.reads():
        read_signals[str(read_record.read_id)] = read_record.signal
    pod5_reader.close()
    print(f"  Loaded {len(read_signals)} read signals", file=sys.stderr)

    # Process BAM
    bam = pysam.AlignmentFile(bam_file, 'rb')

    read_count = 0
    for read in bam.fetch():
        if read.is_unmapped:
            continue

        read_id = read.query_name
        ref_name = read.reference_name

        stride, moves = parse_move_table(read)
        if stride is None:
            continue

        if read_id not in read_signals:
            continue
        signal = read_signals[read_id]

        ss_tag = read.get_tag('ss') if read.has_tag('ss') else 0
        boundaries = get_signal_boundaries(moves, stride, signal_start=ss_tag)

        if ref_name not in sites_by_ref:
            continue

        c_sites = sites_by_ref[ref_name]

        for c_pos in c_sites:
            read_pos = None
            aligned_pairs = read.get_aligned_pairs()
            for qpos, rpos in aligned_pairs:
                if rpos == c_pos and qpos is not None:
                    read_pos = qpos
                    break

            if read_pos is None:
                continue

            sig = extract_signal_at_position(signal, boundaries, read_pos)
            if sig is None or len(sig) == 0:
                continue

            results.append({
                'sample': sample_name,
                'chrom': ref_name,
                'position': c_pos,
                'read_id': read_id,
                'mean_current': np.mean(sig),
                'std_current': np.std(sig),
                'dwell_time': len(sig) / 5000.0,
                'n_samples': len(sig)
            })

        read_count += 1
        if read_count % 1000 == 0:
            print(f"  Processed {read_count} reads, found {len(results)} measurements",
                  file=sys.stderr)

    bam.close()
    print(f"  Total: {read_count} reads, {len(results)} measurements", file=sys.stderr)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Extract signal at cytosine positions (3 samples)')
    parser.add_argument('--control-bam', required=True, help='Control signal-aligned BAM')
    parser.add_argument('--control-pod5', required=True, help='Control POD5 file')
    parser.add_argument('--5mC-bam', dest='mc_bam', required=True, help='5mC signal-aligned BAM')
    parser.add_argument('--5mC-pod5', dest='mc_pod5', required=True, help='5mC POD5 file')
    parser.add_argument('--5hmC-bam', dest='hmc_bam', required=True, help='5hmC signal-aligned BAM')
    parser.add_argument('--5hmC-pod5', dest='hmc_pod5', required=True, help='5hmC POD5 file')
    parser.add_argument('--bed', required=True, help='BED file with cytosine positions')
    parser.add_argument('--output', required=True, help='Output CSV file')

    args = parser.parse_args()

    # Load cytosine positions
    print("Loading cytosine positions...", file=sys.stderr)
    bed_df = load_cytosine_sites(args.bed)
    print(f"  Found {len(bed_df)} cytosine sites across {bed_df['chrom'].nunique()} references",
          file=sys.stderr)

    # Process control sample (canonical C)
    print("\nProcessing control (canonical C) sample...", file=sys.stderr)
    control_df = process_bam_with_pod5(
        args.control_bam, args.control_pod5, bed_df, 'control'
    )

    # Process 5mC sample
    print("\nProcessing 5mC (5-methylcytosine) sample...", file=sys.stderr)
    mc_df = process_bam_with_pod5(
        args.mc_bam, args.mc_pod5, bed_df, '5mC'
    )

    # Process 5hmC sample
    print("\nProcessing 5hmC (5-hydroxymethylcytosine) sample...", file=sys.stderr)
    hmc_df = process_bam_with_pod5(
        args.hmc_bam, args.hmc_pod5, bed_df, '5hmC'
    )

    # Combine results
    combined_df = pd.concat([control_df, mc_df, hmc_df], ignore_index=True)

    # Save to CSV
    combined_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined_df)} measurements to {args.output}", file=sys.stderr)

    # Print summary
    print("\n=== Summary ===", file=sys.stderr)
    summary = combined_df.groupby('sample').agg({
        'mean_current': ['mean', 'std', 'count'],
        'read_id': 'nunique',
        'position': 'nunique'
    })
    print(summary.to_string(), file=sys.stderr)

    # Print per-sample stats
    print("\n=== Per-Sample Current Statistics ===", file=sys.stderr)
    for sample in ['control', '5mC', '5hmC']:
        sample_data = combined_df[combined_df['sample'] == sample]['mean_current']
        if len(sample_data) > 0:
            print(f"{sample:10s}: mean={sample_data.mean():.2f} pA, std={sample_data.std():.2f} pA, n={len(sample_data)}",
                  file=sys.stderr)


if __name__ == '__main__':
    main()
