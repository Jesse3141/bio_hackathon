#!/usr/bin/env python3
"""
Extract per-position current levels from signal-aligned BAM files for ALL positions.

Unlike extract_signal_at_positions.py which extracts only at BED-defined cytosines,
this script extracts signal at EVERY aligned reference position (0-154 for 155bp constructs).

Output format:
    sample, chrom, read_id, position, base, mean_current, std_current, dwell_time, n_samples
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pysam
import pod5
from Bio import SeqIO


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


def load_reference_sequences(fasta_file):
    """Load reference sequences from FASTA file."""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq).upper()
    return sequences


def process_bam_with_pod5_all_positions(bam_file, pod5_file, reference_seqs, sample_name):
    """Extract signal at ALL aligned positions from signal-aligned BAM.

    Args:
        bam_file: Path to signal-aligned BAM
        pod5_file: Path to POD5 file with raw signals
        reference_seqs: Dict of {chrom: sequence} from FASTA
        sample_name: Name for this sample (e.g., 'control', '5mC', '5hmC')

    Returns:
        DataFrame with columns: sample, chrom, read_id, position, base,
                                mean_current, std_current, dwell_time, n_samples
    """
    results = []

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
    skipped_no_moves = 0
    skipped_no_signal = 0

    for read in bam.fetch():
        if read.is_unmapped:
            continue

        read_id = read.query_name
        ref_name = read.reference_name

        # Get move table
        stride, moves = parse_move_table(read)
        if stride is None:
            skipped_no_moves += 1
            continue

        # Get signal for this read
        if read_id not in read_signals:
            skipped_no_signal += 1
            continue
        signal = read_signals[read_id]

        # Get signal boundaries for each base
        ss_tag = read.get_tag('ss') if read.has_tag('ss') else 0
        boundaries = get_signal_boundaries(moves, stride, signal_start=ss_tag)

        # Get reference sequence for this chrom
        if ref_name not in reference_seqs:
            continue
        ref_seq = reference_seqs[ref_name]

        # Get all aligned pairs (query_pos, ref_pos)
        aligned_pairs = read.get_aligned_pairs()

        # Process ALL aligned positions
        for qpos, rpos in aligned_pairs:
            # Skip if either position is None (insertion/deletion)
            if qpos is None or rpos is None:
                continue

            # Skip if reference position out of bounds
            if rpos < 0 or rpos >= len(ref_seq):
                continue

            # Get reference base at this position
            ref_base = ref_seq[rpos]

            # Extract signal at this query position
            sig = extract_signal_at_position(signal, boundaries, qpos)
            if sig is None or len(sig) == 0:
                continue

            results.append({
                'sample': sample_name,
                'chrom': ref_name,
                'read_id': read_id,
                'position': rpos,
                'base': ref_base,
                'mean_current': np.mean(sig),
                'std_current': np.std(sig),
                'dwell_time': len(sig) / 5000.0,  # 5kHz sample rate
                'n_samples': len(sig)
            })

        read_count += 1
        if read_count % 1000 == 0:
            print(f"  Processed {read_count} reads, found {len(results)} measurements",
                  file=sys.stderr)

    bam.close()
    print(f"  Total: {read_count} reads, {len(results)} measurements", file=sys.stderr)
    if skipped_no_moves > 0:
        print(f"  Skipped {skipped_no_moves} reads without move tables", file=sys.stderr)
    if skipped_no_signal > 0:
        print(f"  Skipped {skipped_no_signal} reads without POD5 signal", file=sys.stderr)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Extract signal at ALL positions (not just cytosines) for 3 samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python extract_signal_full_sequence.py \\
        --control-bam output/rep1/control_signal_aligned.bam \\
        --control-pod5 nanopore_ref_data/control_rep1.pod5 \\
        --5mC-bam output/rep1/5mC_signal_aligned.bam \\
        --5mC-pod5 nanopore_ref_data/5mC_rep1.pod5 \\
        --5hmC-bam output/rep1/5hmC_signal_aligned.bam \\
        --5hmC-pod5 nanopore_ref_data/5hmC_rep1.pod5 \\
        --reference nanopore_ref_data/all_5mers.fa \\
        --output output/rep1/signal_full_sequence.csv

Output columns:
    sample:       control, 5mC, or 5hmC
    chrom:        Reference sequence name (e.g., 5mers_rand_ref_adapter_01)
    read_id:      UUID of the nanopore read
    position:     Reference position (0-154 for 155bp constructs)
    base:         Reference nucleotide at this position (A, C, G, T)
    mean_current: Mean pA value at this position
    std_current:  Standard deviation of current
    dwell_time:   Time spent at this position (seconds)
    n_samples:    Number of raw samples averaged
        """
    )
    parser.add_argument('--control-bam', required=True, help='Control signal-aligned BAM')
    parser.add_argument('--control-pod5', required=True, help='Control POD5 file')
    parser.add_argument('--5mC-bam', dest='mc_bam', required=True, help='5mC signal-aligned BAM')
    parser.add_argument('--5mC-pod5', dest='mc_pod5', required=True, help='5mC POD5 file')
    parser.add_argument('--5hmC-bam', dest='hmc_bam', required=False, help='5hmC signal-aligned BAM (optional)')
    parser.add_argument('--5hmC-pod5', dest='hmc_pod5', required=False, help='5hmC POD5 file (optional)')
    parser.add_argument('--reference', required=True, help='Reference FASTA file')
    parser.add_argument('--output', required=True, help='Output CSV file')

    args = parser.parse_args()

    # Load reference sequences
    print("Loading reference sequences...", file=sys.stderr)
    reference_seqs = load_reference_sequences(args.reference)
    print(f"  Loaded {len(reference_seqs)} reference sequences", file=sys.stderr)
    for name, seq in list(reference_seqs.items())[:3]:
        print(f"    {name}: {len(seq)} bp", file=sys.stderr)
    if len(reference_seqs) > 3:
        print(f"    ... and {len(reference_seqs) - 3} more", file=sys.stderr)

    all_dfs = []

    # Process control sample (canonical C)
    print("\n" + "="*60, file=sys.stderr)
    print("Processing control (canonical C) sample...", file=sys.stderr)
    print("="*60, file=sys.stderr)
    control_df = process_bam_with_pod5_all_positions(
        args.control_bam, args.control_pod5, reference_seqs, 'control'
    )
    all_dfs.append(control_df)

    # Process 5mC sample
    print("\n" + "="*60, file=sys.stderr)
    print("Processing 5mC (5-methylcytosine) sample...", file=sys.stderr)
    print("="*60, file=sys.stderr)
    mc_df = process_bam_with_pod5_all_positions(
        args.mc_bam, args.mc_pod5, reference_seqs, '5mC'
    )
    all_dfs.append(mc_df)

    # Process 5hmC sample (optional)
    if args.hmc_bam and args.hmc_pod5:
        print("\n" + "="*60, file=sys.stderr)
        print("Processing 5hmC (5-hydroxymethylcytosine) sample...", file=sys.stderr)
        print("="*60, file=sys.stderr)
        hmc_df = process_bam_with_pod5_all_positions(
            args.hmc_bam, args.hmc_pod5, reference_seqs, '5hmC'
        )
        all_dfs.append(hmc_df)

    # Combine results
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save to CSV
    print(f"\nSaving {len(combined_df)} measurements to {args.output}...", file=sys.stderr)
    combined_df.to_csv(args.output, index=False)
    print("Done!", file=sys.stderr)

    # Print summary
    print("\n" + "="*60, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    print("="*60, file=sys.stderr)

    print("\n--- Per-Sample Statistics ---", file=sys.stderr)
    for sample in combined_df['sample'].unique():
        sample_data = combined_df[combined_df['sample'] == sample]
        print(f"\n{sample}:", file=sys.stderr)
        print(f"  Reads:      {sample_data['read_id'].nunique():,}", file=sys.stderr)
        print(f"  Positions:  {sample_data['position'].nunique()}", file=sys.stderr)
        print(f"  Measurements: {len(sample_data):,}", file=sys.stderr)
        print(f"  Mean current: {sample_data['mean_current'].mean():.2f} ± {sample_data['mean_current'].std():.2f} pA", file=sys.stderr)

    print("\n--- Per-Base Statistics ---", file=sys.stderr)
    base_stats = combined_df.groupby(['sample', 'base'])['mean_current'].agg(['mean', 'std', 'count'])
    print(base_stats.to_string(), file=sys.stderr)

    print("\n--- Per-Position Coverage (first 10 positions) ---", file=sys.stderr)
    pos_coverage = combined_df.groupby(['sample', 'position']).size().unstack(fill_value=0)
    print(pos_coverage.iloc[:, :10].to_string(), file=sys.stderr)


if __name__ == '__main__':
    main()
