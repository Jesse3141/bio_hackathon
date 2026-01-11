import pod5
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os


# --- 1. The Segmentation Logic (Tuned for R10.4 Pores) ---
def fast_segmenter(signal, threshold=9.0, min_duration=2):
    """
    Takes raw ionic current and chops it into segments (events).
    Returns a list of Mean Current values (pA).
    """
    # 1. Light smoothing
    smooth = gaussian_filter1d(signal, sigma=1.0)
    # 2. Calculate Velocity (Derivative)
    grad = np.abs(np.diff(smooth))
    # 3. Find peaks (The "Cliffs")
    jumps = np.where(grad > threshold)[0]
    # Add start and end
    boundaries = np.concatenate(([0], jumps, [len(signal)]))
    # 4. Calculate Mean for each segment
    means = []

    # We zip the boundaries to get (Start, End) pairs
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        duration = end - start
        if duration > min_duration:
            seg_mean = np.mean(signal[start:end])
            means.append(seg_mean)

    return np.array(means)


def get_events_from_pod5(filename, max_reads=None):
    all_events = []

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return []

    print(f"Processing {filename}...")
    with pod5.Reader(filename) as reader:
        for i, read in enumerate(reader.reads()):

            # Check limit
            if max_reads is not None and i >= max_reads:
                break

            raw_signal = read.signal
            # Segment the read
            events = fast_segmenter(raw_signal, threshold=9.0, min_duration=2)

            # Filter for healthy reads
            if len(events) > 50:
                all_events.append(events)

    return all_events
