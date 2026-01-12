# Nanopore Reference Data Summary

## Quick Overview

This is a **ground truth dataset** for 3-way cytosine modification classification (C vs 5mC vs 5hmC) on Oxford Nanopore R10.4.1 chemistry.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE OVERVIEW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RAW DATA (nanopore_ref_data/)                                              │
│  ├── control_rep1.pod5    160,000 reads │ Canonical C                       │
│  ├── 5mC_rep1.pod5        160,000 reads │ 5-methylcytosine                  │
│  └── 5hmC_rep1.pod5       159,943 reads │ 5-hydroxymethylcytosine           │
│                                    ↓                                        │
│                    Dorado basecalling (--emit-moves)                        │
│                    uncalled4 DTW signal alignment                           │
│                                    ↓                                        │
│  ALIGNED DATA (output/rep1/)                                                │
│  ├── control_signal_aligned.bam    65,506 reads │ Signal↔Reference mapping  │
│  ├── 5mC_signal_aligned.bam        33,782 reads │ with move tables          │
│  └── 5hmC_signal_aligned.bam       37,754 reads │                           │
│                                    ↓                                        │
│                    Signal extraction at BED positions                       │
│                                    ↓                                        │
│  HMM TRAINING DATA (output/rep1/)                                           │
│  ├── signal_at_cytosines_3way.csv     1,047,178 measurements (8 C sites)    │
│  ├── signal_full_sequence.csv         15,514,108 measurements (ALL 155 pos) │
│  ├── hmm_3way_circuit_board.json      Gaussian emission parameters          │
│  └── hmm_3way_pomegranate.json        Pomegranate-compatible format         │
│                                                                             │
│  LABELS (nanopore_ref_data/)                                                │
│  └── all_5mers_C_sites.bed            256 cytosine positions                │
│      (modification state = sample condition, not BED label)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Raw Signal Data (POD5 Files)

**Location**: `nanopore_ref_data/`

| File | Size | Reads | Contents |
|------|------|-------|----------|
| `control_rep1.pod5` | 358 MB | 160,000 | Raw pA current at 5 kHz sampling rate |
| `5mC_rep1.pod5` | 345 MB | 160,000 | Same sequences, all C sites are methylated |
| `5hmC_rep1.pod5` | 357 MB | 159,943 | Same sequences, all C sites are hydroxymethylated |

**What's inside**: Raw ionic current measurements from the nanopore. Each read contains:
- `signal`: Array of picoampere (pA) values sampled at 5 kHz
- `read_id`: Unique identifier (UUID)
- No basecalling or alignment information

```python
import pod5
with pod5.Reader("control_rep1.pod5") as reader:
    for read in reader.reads():
        signal = read.signal          # Raw pA values (e.g., 75,000 samples for 15-second read)
        read_id = str(read.read_id)   # UUID
```

---

## 2. Signal-Aligned BAM Files

**Location**: `output/rep1/`

| File | Size | Reads | Contents |
|------|------|-------|----------|
| `control_signal_aligned.bam` | 54 MB | 65,506 | Aligned reads with signal-to-reference mapping |
| `5mC_signal_aligned.bam` | 29 MB | 33,782 | Same, for methylated sample |
| `5hmC_signal_aligned.bam` | 37 MB | 37,754 | Same, for hydroxymethylated sample |

**Why fewer reads?** uncalled4 requires reads long enough to align; ~60% were filtered as "too short".

**What's inside**: Standard BAM alignment plus special tags for signal mapping:

| Tag | Description |
|-----|-------------|
| `mv:B:c` | Move table - maps signal samples to bases (1=move, 0=stay) |
| `ts:i` | Stride - number of signal samples per move table entry (typically 10) |

```python
import pysam
bam = pysam.AlignmentFile("output/rep1/control_signal_aligned.bam", "rb")
for read in bam.fetch():
    moves = read.get_tag("mv")      # Move table array
    stride = read.get_tag("ts")     # Stride value
    ref_pos = read.reference_start  # 0-based position on reference
```

### Understanding the Move Table

The **move table** is the critical data structure that connects raw nanopore signal to the basecalled sequence. It solves the fundamental problem: given 5000 current samples per second, which samples correspond to which DNA base?

**Format**: `mv:B:c,<stride>,<move_0>,<move_1>,...`

- **stride**: Number of raw signal samples per move table entry (typically 6-12)
- **moves**: Binary array where `1` = "new base starts here", `0` = "still on current base"

**Visual Example**:

```
Raw Signal (5 kHz):  [....|....|....|....|....|....|....|....|....]
                         ↑         ↑              ↑         ↑
Move Table:          [1,  0,  0,  1,  0,  0,  0,  1,  0,  0,  1, ...]
                      │           │              │           │
                    Base 1      Base 2         Base 3      Base 4
                   (3 entries)  (4 entries)   (3 entries)

If stride=10: Base 1 = 30 signal samples, Base 2 = 40 samples, etc.
```

**How to Extract Signal for a Base**:

```python
import numpy as np

def get_base_signal(signal, moves, stride, base_index):
    """Extract raw signal samples for a specific base."""
    move_positions = np.where(moves == 1)[0]  # Where each base starts

    start_entry = move_positions[base_index]
    if base_index + 1 < len(move_positions):
        end_entry = move_positions[base_index + 1]
    else:
        end_entry = len(moves)

    # Convert move entries to signal sample indices
    start_sample = start_entry * stride
    end_sample = end_entry * stride

    return signal[start_sample:end_sample]

# The MEAN of this segment is what the HMM observes
base_signal = get_base_signal(signal, moves, stride, base_index=42)
mean_current = np.mean(base_signal)  # This is the HMM observation!
```

**Why This Matters for HMMs**:

The HMM doesn't see raw signal directly. It observes the **mean current per base segment**:

```
Raw Signal → Move Table → Per-base segments → Mean current → HMM observation
   (pA)       (binary)      (pA arrays)          (scalar)      (what we model)
```

Each cytosine position in `signal_at_cytosines_3way.csv` contains this mean current value, computed by segmenting the raw signal using the move table.

**See also**: `visualize_signal_segmentation.py` for a visual demonstration of this segmentation process.

---

## 3. Ground Truth Labels (BED Files)

**Location**: `nanopore_ref_data/`

| File | Positions | Description |
|------|-----------|-------------|
| `all_5mers_C_sites.bed` | 256 | Cytosine positions in the 32 reference sequences |
| `all_5mers_5mC_sites.bed` | 256 | Same positions, labeled as methylated |
| `all_5mers_5hmC_sites.bed` | 256 | Same positions, labeled as hydroxymethylated |

**Key insight**: The BED files define WHERE cytosines are (same 256 positions in all files). The modification state comes from WHICH POD5 FILE the read came from:

| Sample File | Modification at all 256 positions |
|-------------|-----------------------------------|
| `control_rep1.pod5` | Canonical C (unmodified) |
| `5mC_rep1.pod5` | 5-methylcytosine |
| `5hmC_rep1.pod5` | 5-hydroxymethylcytosine |

**BED format**:
```
chrom                        start  end  label  score  strand
5mers_rand_ref_adapter_01    38     39   -      0      +
5mers_rand_ref_adapter_01    50     51   -      0      +
...
```

**Positions per construct**: 38, 50, 62, 74, 86, 98, 110, 122 (8 cytosines × 32 constructs = 256 total)

### Score Column (Sequence Context)

The `score` column (5th field) indicates the sequence context around each cytosine:

| Score | Context | Count | Definition |
|-------|---------|-------|------------|
| **0** | Non-CpG | 156 | Cytosine NOT followed by G |
| **1** | CpG | 64 | Cytosine followed by G (CpG dinucleotide) |
| **2** | Homopolymer | 36 | Non-CpG cytosine adjacent to another C (CC run) |

**Examples**:
```
Score 0: TTCCA, CACCG, CGCAC  → C not followed by G, not in CC run
Score 1: CACGC, CGCGC, GCCGA  → C followed by G (CpG site)
Score 2: TTCCC, CCCAA, TCCCG  → C next to another C (homopolymer)
```

**Why this matters**:
- **CpG (score 1)**: Primary biological context for mammalian methylation
- **Homopolymer (score 2)**: Harder to align, noisier signal due to ambiguous segmentation
- **Non-CpG (score 0)**: Cleanest context for signal analysis

Use this to stratify HMM training/evaluation by context difficulty.

---

## 4. HMM Training Data

**Location**: `output/rep1/`

### 4a. Extracted Signal Values (CSV)

**File**: `signal_at_cytosines_3way.csv` (115 MB, 1,047,178 rows)

| Column | Description |
|--------|-------------|
| `sample` | Source: "control", "5mC", or "5hmC" |
| `chrom` | Reference sequence name |
| `position` | Cytosine position (38, 50, 62, 74, 86, 98, 110, 122) |
| `read_id` | UUID linking back to POD5/BAM |
| `mean_current` | Mean pA value at this position (**HMM observation**) |
| `std_current` | Standard deviation of current |
| `dwell_time` | Time spent at this position (seconds) |
| `n_samples` | Number of raw samples averaged |

```csv
sample,chrom,position,read_id,mean_current,std_current,dwell_time,n_samples
control,5mers_rand_ref_adapter_01,98,b1b7b72a-...,814.0,70.7,0.0012,6
5mC,5mers_rand_ref_adapter_01,98,a2c3d4e5-...,842.3,68.2,0.0014,7
5hmC,5mers_rand_ref_adapter_01,98,f6g7h8i9-...,821.5,72.1,0.0011,5
```

### 4b. HMM Emission Parameters (JSON)

**File**: `hmm_3way_circuit_board.json` (3.2 KB)

Pre-computed Gaussian parameters for each modification state at each position:

```json
{
  "distributions": [
    {
      "position": 38,
      "C":   {"mean": 866.9, "std": 106.2, "count": 64522},
      "mC":  {"mean": 893.6, "std": 108.2, "count": 33282},
      "hmC": {"mean": 856.9, "std": 110.0, "count": 33811}
    },
    ...
  ]
}
```

**This is what the HMM needs**: Gaussian distributions N(μ, σ²) for each (position, modification_state) pair.

### 4c. Full-Sequence Signal Data (ALL Positions)

**File**: `signal_full_sequence.csv` (1.7 GB, 15,514,108 rows)

Unlike `signal_at_cytosines_3way.csv` which extracts only at 8 cytosine positions, this file contains signal measurements at **ALL 155 reference positions** for every aligned read.

| Column | Description |
|--------|-------------|
| `sample` | Source: "control", "5mC", or "5hmC" |
| `chrom` | Reference sequence name |
| `read_id` | UUID linking back to POD5/BAM |
| `position` | Reference position (0-154, ALL positions) |
| `base` | Reference nucleotide at this position (A, C, G, T) |
| `mean_current` | Mean pA value at this position |
| `std_current` | Standard deviation of current |
| `dwell_time` | Time spent at this position (seconds) |
| `n_samples` | Number of raw samples averaged |

**Sample counts:**

| Sample | Reads | Measurements | Mean Current |
|--------|-------|--------------|--------------|
| control | 62,014 | 7,478,101 | 798.03 ± 111.23 pA |
| 5mC | 33,298 | 3,918,378 | 823.96 ± 116.03 pA |
| 5hmC | 29,943 | 4,117,629 | 815.85 ± 116.10 pA |

**Per-base current differences:**

| Base | Control (pA) | 5mC (pA) | 5hmC (pA) | Δ 5mC-C | Δ 5hmC-C |
|------|-------------|----------|-----------|---------|----------|
| A | 794.1 | 819.5 | 813.0 | +25.4 | +18.9 |
| C | 800.7 | 829.6 | 817.0 | +28.9 | +16.3 |
| G | 803.0 | 829.6 | 820.3 | +26.6 | +17.3 |
| T | 795.4 | 818.2 | 814.4 | +22.8 | +19.0 |

**Use case**: This data enables training HMMs that model the ENTIRE read sequence, not just cytosine fork positions. Essential for:
- K-mer context modeling (current depends on surrounding bases)
- Position-specific emission models
- Training linear-chain HMMs over full sequences

**Generated by**: `extract_signal_full_sequence.py`

```bash
python extract_signal_full_sequence.py \
    --control-bam output/rep1/control_signal_aligned.bam \
    --control-pod5 nanopore_ref_data/control_rep1.pod5 \
    --5mC-bam output/rep1/5mC_signal_aligned.bam \
    --5mC-pod5 nanopore_ref_data/5mC_rep1.pod5 \
    --5hmC-bam output/rep1/5hmC_signal_aligned.bam \
    --5hmC-pod5 nanopore_ref_data/5hmC_rep1.pod5 \
    --reference nanopore_ref_data/all_5mers.fa \
    --output output/rep1/signal_full_sequence.csv
```

---

## 5. What the HMM Model Needs

### Input for Training

1. **Emission distributions** (already computed):
   - `output/rep1/hmm_3way_circuit_board.json` → Gaussian parameters per state
   - Or use `output/rep1/signal_at_cytosines_3way.csv` to compute your own

2. **Transition probabilities** (to define):
   - Self-loop probabilities for each state
   - Transition probabilities between positions

### Input for Inference (Classifying New Reads)

1. **Signal-aligned BAM** with move tables (`mv` tag)
2. **POD5 file** with raw signal
3. **Trained HMM model** with emission/transition parameters

### Signal Characteristics (Summary)

| Modification | Mean Current | Δ vs C | Count |
|--------------|--------------|--------|-------|
| C (canonical) | 800.5 pA | — | 513,290 |
| 5mC | 830.7 pA | +30.2 pA | 264,300 |
| 5hmC | 809.4 pA | +8.9 pA | 269,588 |

**Ordering**: 5mC > 5hmC > C (methylation increases current)

---

## 6. Reference Sequences

**File**: `all_5mers.fa` (5.8 KB)

32 synthetic constructs, each 155 bp with the same structure:

```
5'─[ADAPTER 28bp]─[VARIABLE 99bp]─[ADAPTER 28bp]─3'
```

Cytosine positions (0-indexed): 38, 50, 62, 74, 86, 98, 110, 122

Each construct covers different 5-mer contexts around the cytosine, enabling context-dependent analysis.

---

## 7. File Dependency Graph

```
nanopore_ref_data/
├── all_5mers.fa              ← Reference sequences (input to alignment)
├── all_5mers_C_sites.bed     ← Cytosine positions (input to extraction)
├── control_rep1.pod5         ← Raw signal (input to basecalling)
├── 5mC_rep1.pod5
└── 5hmC_rep1.pod5

output/rep1/
├── *_with_moves.bam          ← Dorado output (intermediate)
├── *_signal_aligned.bam      ← uncalled4 output (for signal extraction)
├── signal_at_cytosines_3way.csv  ← Cytosine-only observations (8 positions)
├── signal_full_sequence.csv      ← ALL positions (0-154), 15.5M rows
├── hmm_3way_circuit_board.json   ← HMM emission parameters
└── hmm_3way_pomegranate.json     ← Alternative format
```

---

## 8. Sequencing Platform Details

| Parameter | Value |
|-----------|-------|
| Chemistry | R10.4.1 E8.2 |
| Flowcell | FLO-PRO114M (PromethION) |
| Kit | SQK-LSK114 (Ligation) |
| Speed | 400 bases/second |
| Sample Rate | 5 kHz |
| Basecaller | Dorado 1.3.0 (HAC model) |

---

## 9. Download Links

Data is publicly available from ONT Open Data:

```
https://42basepairs.com/download/s3/ont-open-data/modbase-validation_2024.10/references/all_5mers.fa
https://42basepairs.com/download/s3/ont-open-data/modbase-validation_2024.10/references/all_5mers_C_sites.bed
https://42basepairs.com/download/s3/ont-open-data/modbase-validation_2024.10/references/all_5mers_5mC_sites.bed
https://42basepairs.com/download/s3/ont-open-data/modbase-validation_2024.10/references/all_5mers_5hmC_sites.bed
https://42basepairs.com/download/s3/ont-open-data/modbase-validation_2024.10/subset/control_rep1.pod5
https://42basepairs.com/download/s3/ont-open-data/modbase-validation_2024.10/subset/5mC_rep1.pod5
https://42basepairs.com/download/s3/ont-open-data/modbase-validation_2024.10/subset/5hmC_rep1.pod5
```

---

## Related Documentation

- [Signal Alignment Setup Guide](../SIGNAL_ALIGNMENT_SETUP.md) - How the pipeline was run
- [HMM Model Building Guide](kmer_models/HMM_MODEL_BUILDING_GUIDE.md) - Building HMMs from k-mer tables
- [Project CLAUDE.md](../CLAUDE.md) - Project overview
