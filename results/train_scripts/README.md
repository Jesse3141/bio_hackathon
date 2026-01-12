# Training Scripts for Methylation HMM

Data loading and preprocessing utilities for HMM-based methylation classification.

## Overview

This module provides tools to:
1. Load signal data from nanopore sequencing
2. Filter by adapter (sequence) and sample type
3. Pivot data to HMM training format
4. Annotate positions with cytosine context (CpG, non-CpG, homopolymer)

Supports both **binary** (C vs 5mC) and **3-way** (C vs 5mC vs 5hmC) classification.

---

## Quick Start

```python
from results.train_scripts import prepare_training_data

# Binary classification on adapter_01
df = prepare_training_data(
    adapter="5mers_rand_ref_adapter_01",
    classification="binary"
)

# 3-way classification on all adapters
df = prepare_training_data(classification="3way")
```

---

## Data Processing Options

### 1. Filter by Adapter (Sequence)

The dataset contains 32 synthetic DNA constructs (`adapter_01` through `adapter_32`). Each has different 5-mer contexts around the 8 cytosine positions.

```python
from results.train_scripts import load_signal_data, ADAPTERS

# Single adapter (recommended for initial experiments)
df = load_signal_data(adapters=["5mers_rand_ref_adapter_01"])

# Multiple adapters
df = load_signal_data(adapters=["5mers_rand_ref_adapter_01", "5mers_rand_ref_adapter_02"])

# All adapters (default)
df = load_signal_data()  # ~1M measurements

# List available adapters
print(ADAPTERS)  # ['5mers_rand_ref_adapter_01', ..., '5mers_rand_ref_adapter_32']
```

### 2. Filter by Sample Type (Classification Mode)

| Mode | Samples | Use Case |
|------|---------|----------|
| Binary | `control`, `5mC` | Standard methylation detection |
| 3-way | `control`, `5mC`, `5hmC` | Distinguish hydroxymethylation |

```python
from results.train_scripts import load_signal_data, BINARY_SAMPLES, THREE_WAY_SAMPLES

# Binary classification (C vs 5mC)
df = load_signal_data(samples=["control", "5mC"])
# or
df = load_signal_data(samples=BINARY_SAMPLES)

# 3-way classification (C vs 5mC vs 5hmC)
df = load_signal_data(samples=["control", "5mC", "5hmC"])
# or
df = load_signal_data(samples=THREE_WAY_SAMPLES)
```

### 3. Filter by Cytosine Context (Site Type)

Each cytosine position has a context score from the BED file:

| Score | Name | Description | Count |
|-------|------|-------------|-------|
| 0 | `non_cpg` | C not followed by G | 156 sites |
| 1 | `cpg` | CpG dinucleotide | 64 sites |
| 2 | `homopolymer` | CC run (adjacent cytosines) | 36 sites |

```python
from results.train_scripts import load_bed_with_site_types, get_positions_by_site_type

# Load site type annotations
bed_df = load_bed_with_site_types()

# Get positions by site type for a specific adapter
site_positions = get_positions_by_site_type(bed_df, "5mers_rand_ref_adapter_01")
# Returns: {0: [38, 50, 62, 110, 122], 1: [86, 98], 2: [74]}

# Filter training data to specific site types
from results.train_scripts import prepare_training_data

df = prepare_training_data(adapter="5mers_rand_ref_adapter_01", classification="binary")

# Get only CpG positions (site_type == 1)
cpg_positions = site_positions[1]  # [86, 98]
cpg_cols = [str(p) for p in cpg_positions]
df_cpg = df[["sample", "chrom", "read_id"] + cpg_cols]
```

### 4. Pivot to Training Format

The raw signal data is in **long format** (one row per position per read). HMM training requires **wide format** (one row per read with all positions as columns).

```python
from results.train_scripts import load_signal_data, pivot_to_training_format

# Long format: 27,170 rows for adapter_01 binary
signal_df = load_signal_data(
    adapters=["5mers_rand_ref_adapter_01"],
    samples=["control", "5mC"]
)
print(signal_df.shape)  # (27170, 8)

# Wide format: 3,468 reads
training_df = pivot_to_training_format(signal_df)
print(training_df.shape)  # (3468, 11)
print(training_df.columns.tolist())
# ['sample', 'chrom', 'read_id', '38', '50', '62', '74', '86', '98', '110', '122']
```

### 5. Add Site Type Annotations

Add per-position site type columns to training data:

```python
from results.train_scripts import prepare_training_data, add_site_types

# prepare_training_data automatically adds site types
df = prepare_training_data(adapter="5mers_rand_ref_adapter_01", classification="binary")

# Site type columns added:
# site_type_38, site_type_50, ..., site_type_122
print(df["site_type_38"].unique())  # [0] for adapter_01
print(df["site_type_86"].unique())  # [1] for adapter_01 (CpG)
```

---

## Data Summaries

### Site Type Distribution

```python
from results.train_scripts import get_site_type_summary

summary = get_site_type_summary()
print(summary)
#  site_type        name  count  positions
#          0     non_cpg    156  [38, 50, 62, 74, 86, 98, 110, 122]
#          1         cpg     64  [38, 50, 62, 74, 86, 98, 110, 122]
#          2 homopolymer     36  [38, 50, 62, 74, 86, 98, 110, 122]
```

### Read Counts per Sample

```python
from results.train_scripts import get_data_summary

# For a specific adapter
summary = get_data_summary(adapters=["5mers_rand_ref_adapter_01"])
print(summary)
#  sample                     chrom  n_reads
#    5hmC 5mers_rand_ref_adapter_01      994
#     5mC 5mers_rand_ref_adapter_01     1217
# control 5mers_rand_ref_adapter_01     2251
```

---

## File Paths

Default paths are configured in `config.py`:

| Variable | Path | Description |
|----------|------|-------------|
| `SIGNAL_CSV` | `output/rep1/signal_at_cytosines_3way.csv` | Raw signal measurements |
| `BED_FILE` | `nanopore_ref_data/all_5mers_C_sites.bed` | Cytosine positions with site types |
| `MODEL_OUTPUT_DIR` | `results/models/` | Saved model directory |

Override defaults by passing paths directly:

```python
df = load_signal_data(csv_path="/custom/path/signal.csv")
bed = load_bed_with_site_types(bed_path="/custom/path/sites.bed")
```

---

## Constants

```python
from results.train_scripts import (
    POSITIONS,        # [38, 50, 62, 74, 86, 98, 110, 122]
    POSITION_COLS,    # ['38', '50', '62', '74', '86', '98', '110', '122']
    ADAPTERS,         # ['5mers_rand_ref_adapter_01', ..., '5mers_rand_ref_adapter_32']
    DEFAULT_ADAPTER,  # '5mers_rand_ref_adapter_01'
    SITE_TYPES,       # {0: 'non_cpg', 1: 'cpg', 2: 'homopolymer'}
    BINARY_SAMPLES,   # ['control', '5mC']
    THREE_WAY_SAMPLES # ['control', '5mC', '5hmC']
)
```

---

## Example Workflows

### Train Binary Classifier on Single Adapter

```python
from results.train_scripts import prepare_training_data

# Prepare data
df = prepare_training_data(
    adapter="5mers_rand_ref_adapter_01",
    classification="binary"
)

# Split into train/test
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["sample"])

# Train (using existing classifier)
from methylation_hmm.simplified_pipeline import SimplifiedMethylationClassifier
# ... training code
```

### Evaluate by Site Type

```python
from results.train_scripts import prepare_training_data, get_positions_by_site_type, load_bed_with_site_types

df = prepare_training_data(adapter="5mers_rand_ref_adapter_01", classification="binary")
bed_df = load_bed_with_site_types()
site_positions = get_positions_by_site_type(bed_df, "5mers_rand_ref_adapter_01")

# Evaluate accuracy per site type
for site_type, positions in site_positions.items():
    pos_cols = [str(p) for p in positions]
    # ... evaluate model on these positions only
    print(f"Site type {site_type}: {len(positions)} positions")
```

### Compare Adapters

```python
from results.train_scripts import get_data_summary

# Get read counts for all adapters
summary = get_data_summary()
print(summary.groupby("sample")["n_reads"].sum())
```

---

## API Reference

| Function | Description |
|----------|-------------|
| `load_bed_with_site_types(bed_path)` | Load BED file with site type scores |
| `load_signal_data(csv_path, adapters, samples)` | Load raw signal CSV with filtering |
| `pivot_to_training_format(signal_df)` | Convert long → wide format |
| `add_site_types(training_df, bed_df)` | Add site_type_XX columns |
| `get_positions_by_site_type(bed_df, chrom)` | Get positions grouped by context |
| `prepare_training_data(signal_csv, bed_path, adapter, classification)` | High-level data prep |
| `get_site_type_summary(bed_path)` | Summary of site types |
| `get_data_summary(csv_path, adapters)` | Summary of read counts |
