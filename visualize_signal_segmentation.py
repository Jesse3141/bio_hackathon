#!/usr/bin/env python3
"""
Visualize how the move table segments raw nanopore signal into per-base segments.

This script demonstrates the core data structure that connects raw signal to bases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysam
import pod5
from pathlib import Path
import argparse


def parse_move_table(read):
    """Parse the move table (mv tag) from a BAM read.

    The move table format is: mv:B:c,<stride>,<moves...>
    - stride: number of signal samples per move table entry
    - moves: array of 0s and 1s where 1 = "move to next base"

    Returns:
        stride: int, signal samples per move entry
        moves: numpy array of 0s and 1s
    """
    mv_tag = read.get_tag('mv') if read.has_tag('mv') else None
    if mv_tag is None:
        return None, None

    stride = mv_tag[0]
    moves = np.array(mv_tag[1:], dtype=np.int8)
    return stride, moves


def get_signal_segments(moves, stride, total_signal_length, signal_start=0):
    """Convert move table to signal segment boundaries for each base.

    Args:
        moves: binary array where 1 = new base starts
        stride: signal samples per move entry
        total_signal_length: length of the raw signal array
        signal_start: offset into signal where alignment starts (ss tag)

    Returns:
        List of (start, end) tuples for each base's signal segment
    """
    segments = []
    move_positions = np.where(moves == 1)[0]

    for i, pos in enumerate(move_positions):
        start = signal_start + pos * stride

        # End is where the next base starts (or end of moves)
        if i + 1 < len(move_positions):
            end = signal_start + move_positions[i + 1] * stride
        else:
            # Last base - extend to end of move table
            end = signal_start + len(moves) * stride

        # Clip to actual signal length
        start = min(start, total_signal_length)
        end = min(end, total_signal_length)

        if start < end:
            segments.append((start, end))

    return segments


def plot_signal_segmentation(signal, segments, sequence, output_file=None,
                             start_base=0, num_bases=20, title=None):
    """Plot raw signal with base segmentation overlay.

    Args:
        signal: raw signal array (pA values)
        segments: list of (start, end) tuples for each base
        sequence: basecalled sequence string
        output_file: path to save figure (optional)
        start_base: first base to show
        num_bases: number of bases to display
        title: plot title
    """
    # Subset to requested bases
    end_base = min(start_base + num_bases, len(segments))
    segments_subset = segments[start_base:end_base]
    sequence_subset = sequence[start_base:end_base] if sequence else None

    if not segments_subset:
        print("No segments to plot")
        return

    # Get signal range
    sig_start = segments_subset[0][0]
    sig_end = segments_subset[-1][1]
    signal_subset = signal[sig_start:sig_end]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    # Top plot: Raw signal with segment boundaries
    ax1 = axes[0]
    x = np.arange(sig_start, sig_end)
    ax1.plot(x, signal_subset, 'b-', linewidth=0.5, alpha=0.7, label='Raw signal')

    # Add segment boundaries and mean lines
    colors = plt.cm.Set3(np.linspace(0, 1, num_bases))
    mean_currents = []

    for i, (start, end) in enumerate(segments_subset):
        seg_signal = signal[start:end]
        mean_current = np.mean(seg_signal)
        mean_currents.append(mean_current)

        # Shade segment
        ax1.axvspan(start, end, alpha=0.3, color=colors[i])

        # Draw mean line
        ax1.hlines(mean_current, start, end, colors='red', linewidths=2,
                   label='Mean current' if i == 0 else None)

        # Add vertical boundary
        ax1.axvline(start, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        # Label with base
        if sequence_subset:
            base = sequence_subset[i]
            ax1.text((start + end) / 2, ax1.get_ylim()[1] * 0.95, base,
                    ha='center', va='top', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Signal Sample Index')
    ax1.set_ylabel('Current (pA)')
    ax1.set_title(title or f'Raw Signal Segmented by Move Table (bases {start_base}-{end_base-1})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Bar chart of mean currents per base
    ax2 = axes[1]
    base_positions = np.arange(len(mean_currents))
    bars = ax2.bar(base_positions, mean_currents, color=colors[:len(mean_currents)],
                   edgecolor='black', linewidth=0.5)

    # Add base labels
    if sequence_subset:
        ax2.set_xticks(base_positions)
        ax2.set_xticklabels(list(sequence_subset), fontsize=10, fontweight='bold')

    ax2.set_xlabel('Base Position')
    ax2.set_ylabel('Mean Current (pA)')
    ax2.set_title('Mean Current per Base Segment')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_currents)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")

    plt.show()

    return mean_currents


def main():
    parser = argparse.ArgumentParser(
        description='Visualize signal segmentation from move table'
    )
    parser.add_argument('--bam', default='output/rep1/control_signal_aligned.bam',
                        help='Signal-aligned BAM file')
    parser.add_argument('--pod5', default='nanopore_ref_data/control_rep1.pod5',
                        help='POD5 file with raw signal')
    parser.add_argument('--read-index', type=int, default=0,
                        help='Which read to visualize (0-indexed)')
    parser.add_argument('--start-base', type=int, default=30,
                        help='First base to display')
    parser.add_argument('--num-bases', type=int, default=25,
                        help='Number of bases to display')
    parser.add_argument('--output', default='output/signal_segmentation.png',
                        help='Output plot file')

    args = parser.parse_args()

    # Check files exist
    bam_path = Path(args.bam)
    pod5_path = Path(args.pod5)

    if not bam_path.exists():
        print(f"BAM file not found: {bam_path}")
        return
    if not pod5_path.exists():
        print(f"POD5 file not found: {pod5_path}")
        return

    # Load POD5 signals
    print(f"Loading signals from {pod5_path}...")
    read_signals = {}
    with pod5.Reader(str(pod5_path)) as reader:
        for i, read_record in enumerate(reader.reads()):
            read_signals[str(read_record.read_id)] = read_record.signal
            if i >= 10000:  # Limit for memory
                break
    print(f"  Loaded {len(read_signals)} read signals")

    # Find a good read from BAM
    print(f"Scanning BAM for suitable read...")
    bam = pysam.AlignmentFile(str(bam_path), 'rb')

    read_count = 0
    target_read = None

    for read in bam.fetch():
        if read.is_unmapped:
            continue
        if not read.has_tag('mv'):
            continue

        read_id = read.query_name
        if read_id not in read_signals:
            continue

        if read_count == args.read_index:
            target_read = read
            break
        read_count += 1

    bam.close()

    if target_read is None:
        print(f"Could not find read at index {args.read_index}")
        return

    # Extract data
    read_id = target_read.query_name
    signal = read_signals[read_id]
    sequence = target_read.query_sequence
    stride, moves = parse_move_table(target_read)

    # Get signal start offset (ss tag from uncalled4)
    signal_start = target_read.get_tag('ss') if target_read.has_tag('ss') else 0

    print(f"\n=== Read Information ===")
    print(f"Read ID: {read_id}")
    print(f"Reference: {target_read.reference_name}")
    print(f"Position: {target_read.reference_start}")
    print(f"Sequence length: {len(sequence)} bases")
    print(f"Signal length: {len(signal):,} samples")
    print(f"Stride: {stride}")
    print(f"Move table length: {len(moves):,}")
    print(f"Signal start offset: {signal_start}")

    # Get segments
    segments = get_signal_segments(moves, stride, len(signal), signal_start)
    print(f"Number of base segments: {len(segments)}")

    # Show segment statistics
    segment_lengths = [end - start for start, end in segments]
    print(f"\nSegment statistics:")
    print(f"  Mean samples/base: {np.mean(segment_lengths):.1f}")
    print(f"  Min samples/base: {np.min(segment_lengths)}")
    print(f"  Max samples/base: {np.max(segment_lengths)}")
    print(f"  Mean dwell time: {np.mean(segment_lengths) / 5000 * 1000:.2f} ms")

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Plot
    print(f"\nPlotting bases {args.start_base} to {args.start_base + args.num_bases - 1}...")
    title = f"Signal Segmentation: {target_read.reference_name} (read {read_id[:8]}...)"

    mean_currents = plot_signal_segmentation(
        signal, segments, sequence,
        output_file=args.output,
        start_base=args.start_base,
        num_bases=args.num_bases,
        title=title
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
