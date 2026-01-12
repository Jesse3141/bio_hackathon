#!/usr/bin/env python3
"""
Extract per-position current levels from signal-aligned BAM files.

This script reads signal-aligned BAMs produced by uncalled4 and extracts
current level statistics at cytosine positions defined in BED files.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pysam
import pod5


def parse_move_table(read):
    """Parse the move table (mv tag) from a BAM read.

    The mv tag format is: mv:B:c,stride,move1,move2,...
    where stride is the number of signal samples per move.
    Each move value indicates how many bases were emitted (0 or 1 typically).
    """
    mv_tag = read.get_tag('mv') if read.has_tag('mv') else None
    ts_tag = read.get_tag('ts') if read.has_tag('ts') else None

    if mv_tag is None:
        return None, None

    # mv_tag is a list where first element is stride, rest are moves
    stride = mv_tag[0]
    moves = np.array(mv_tag[1:], dtype=np.int8)

    return stride, moves


def get_signal_boundaries(moves, stride, signal_start=0):
    """Convert move table to signal sample boundaries for each base.

    Returns array of (start_sample, end_sample) for each base position.
    """
    boundaries = []
    current_sample = signal_start

    for i, move in enumerate(moves):
        if move == 1:
            # This position emitted a base
            start = current_sample
            # Find where this base ends (next move=1 or end)
            j = i + 1
            while j < len(moves) and moves[j] == 0:
                j += 1
            end = signal_start + j * stride
            boundaries.append((start, end))
        current_sample += stride

    return boundaries


def extract_signal_at_position(signal, boundaries, base_position):
    """Extract signal values for a specific base position.

    Args:
        signal: Full signal array for the read
        boundaries: List of (start, end) sample boundaries per base
        base_position: Position in the sequence (0-indexed from read start)

    Returns:
        Array of signal values at that position, or None if out of bounds.
    """
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
    """Extract signal at cytosine positions from signal-aligned BAM.

    Args:
        bam_file: Path to signal-aligned BAM
        pod5_file: Path to POD5 file with raw signals
        bed_df: DataFrame with cytosine positions
        sample_name: Name for this sample (e.g., 'control' or '5mC')

    Returns:
        DataFrame with columns: sample, chrom, position, read_id,
                                 mean_current, std_current, dwell_time, n_samples
    """
    results = []

    # Create lookup of cytosine positions per reference
    sites_by_ref = bed_df.groupby('chrom')['start'].apply(set).to_dict()

    # Load POD5 for signal data
    pod5_reader = pod5.Reader(pod5_file)
    read_signals = {}
    for read_record in pod5_reader.reads():
        read_signals[str(read_record.read_id)] = read_record.signal
    pod5_reader.close()

    # Process BAM
    bam = pysam.AlignmentFile(bam_file, 'rb')

    read_count = 0
    for read in bam.fetch():
        if read.is_unmapped:
            continue

        read_id = read.query_name
        ref_name = read.reference_name
        ref_start = read.reference_start  # 0-indexed

        # Get move table
        stride, moves = parse_move_table(read)
        if stride is None:
            continue

        # Get signal for this read
        if read_id not in read_signals:
            continue
        signal = read_signals[read_id]

        # Get signal boundaries for each base
        ss_tag = read.get_tag('ss') if read.has_tag('ss') else 0
        boundaries = get_signal_boundaries(moves, stride, signal_start=ss_tag)

        # Get cytosine sites for this reference
        if ref_name not in sites_by_ref:
            continue

        c_sites = sites_by_ref[ref_name]

        # For each cytosine position, extract signal
        for c_pos in c_sites:
            # Convert reference position to read position
            # Need to account for alignment (CIGAR)
            read_pos = None

            # Simple case: no indels, just offset
            aligned_pairs = read.get_aligned_pairs()
            for qpos, rpos in aligned_pairs:
                if rpos == c_pos and qpos is not None:
                    read_pos = qpos
                    break

            if read_pos is None:
                continue

            # Extract signal at this position
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
                'dwell_time': len(sig) / 5000.0,  # 5kHz sample rate
                'n_samples': len(sig)
            })

        read_count += 1
        if read_count % 500 == 0:
            print(f"  Processed {read_count} reads, found {len(results)} measurements",
                  file=sys.stderr)

    bam.close()
    print(f"  Total: {read_count} reads, {len(results)} measurements", file=sys.stderr)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Extract signal at cytosine positions')
    parser.add_argument('--control-bam', required=True, help='Control signal-aligned BAM')
    parser.add_argument('--control-pod5', required=True, help='Control POD5 file')
    parser.add_argument('--modified-bam', required=True, help='Modified signal-aligned BAM')
    parser.add_argument('--modified-pod5', required=True, help='Modified POD5 file')
    parser.add_argument('--bed', required=True, help='BED file with cytosine positions')
    parser.add_argument('--output', required=True, help='Output CSV file')

    args = parser.parse_args()

    # Load cytosine positions
    print("Loading cytosine positions...", file=sys.stderr)
    bed_df = load_cytosine_sites(args.bed)
    print(f"  Found {len(bed_df)} cytosine sites across {bed_df['chrom'].nunique()} references",
          file=sys.stderr)

    # Process control sample
    print("\nProcessing control sample...", file=sys.stderr)
    control_df = process_bam_with_pod5(
        args.control_bam, args.control_pod5, bed_df, 'control'
    )

    # Process modified sample
    print("\nProcessing modified sample...", file=sys.stderr)
    modified_df = process_bam_with_pod5(
        args.modified_bam, args.modified_pod5, bed_df, 'modified'
    )

    # Combine results
    combined_df = pd.concat([control_df, modified_df], ignore_index=True)

    # Save to CSV
    combined_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(combined_df)} measurements to {args.output}", file=sys.stderr)

    # Print summary
    print("\n=== Summary ===", file=sys.stderr)
    summary = combined_df.groupby('sample').agg({
        'mean_current': ['mean', 'std'],
        'read_id': 'nunique',
        'position': 'nunique'
    })
    print(summary.to_string(), file=sys.stderr)


if __name__ == '__main__':
    main()
