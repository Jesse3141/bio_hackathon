# Signal Alignment Pipeline Setup Guide

> **Related Documentation**: [README.md](README.md) | [RESULTS.md](RESULTS.md) | [nanopore_ref_data/DATA_SUMMARY.md](nanopore_ref_data/DATA_SUMMARY.md) | [methylation_hmm/PLAN.md](methylation_hmm/PLAN.md)

## Overview

This guide sets up the pipeline to create **signal-aligned BAM files** from nanopore POD5 data. These files contain move tables (`mv` tags) that map raw signal samples to basecalled nucleotides - essential for HMM-based modification detection.

```
POD5 (raw signal) → Dorado (basecall + emit-moves) → BAM with mv tags → uncalled4 (DTW alignment) → Signal-aligned BAM
```

---

## Current Installation Status

| Component | Status | Location |
|-----------|--------|----------|
| Dorado basecaller | ✅ Installed | `~/software/dorado-1.3.0-linux-x64/bin/dorado` |
| HAC model | ✅ Downloaded | `~/software/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0` |
| Methylation model | ✅ Downloaded | `~/software/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v2` |
| Conda environment | ✅ Created | `nanopore` (Python 3.10) |
| Python packages | ✅ Installed | uncalled4, pod5, pysam, pandas, numpy, scipy |
| Samtools | ✅ Installed | v1.21 in conda env |
| NVIDIA GPU | ✅ Detected | RTX 3080 |

**Quick start:**
```bash
conda activate nanopore
~/software/dorado-1.3.0-linux-x64/bin/dorado --version
```

---

## Available Data

### Reference Data (`nanopore_ref_data/`)

| File | Description |
|------|-------------|
| `all_5mers.fa` | Reference FASTA - 32 synthetic 155bp constructs |
| `all_5mers_C_sites.bed` | Canonical cytosine positions (256 sites) |
| `all_5mers_5mC_sites.bed` | Methylated cytosine positions (256 sites) |
| `control_rep1.bam` | Pre-aligned reads - canonical (unmodified) |
| `5hmC_rep1.bam` | Pre-aligned reads - hydroxymethylated |
| `DATA_SUMMARY.md` | Full documentation of the truth set |

### POD5 Data (`filtered_pod_files/`)

| File | Description |
|------|-------------|
| `control1_filtered_adapter_01.pod5` | Raw signal - canonical cytosines |
| `5mc_filtered_adapter_01.pod5` | Raw signal - 5-methylcytosines |

---

## Prerequisites

- Linux x64 (Ubuntu/Debian recommended)
- NVIDIA GPU with CUDA (optional but ~50x faster)
- ~10GB disk space for models
- conda/mamba environment

---

## Step 1: Install System Dependencies

```bash
# For building Python packages with C extensions
sudo apt-get update
sudo apt-get install -y zlib1g-dev build-essential

# For GPU acceleration (if using NVIDIA GPU)
# Ensure CUDA toolkit is installed: https://developer.nvidia.com/cuda-downloads
nvidia-smi  # Verify GPU is detected
```

---

## Step 2: Install Dorado Basecaller

Dorado is ONT's GPU-accelerated basecaller.

```bash
# Create software directory
mkdir -p ~/software && cd ~/software

# Download latest dorado (check https://github.com/nanoporetech/dorado/releases)
wget https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.3.0-linux-x64.tar.gz

# Extract
tar -xzf dorado-1.3.0-linux-x64.tar.gz

# Add to PATH (add to ~/.bashrc for persistence)
echo 'export PATH=$PATH:~/software/dorado-1.3.0-linux-x64/bin' >> ~/.bashrc
source ~/.bashrc

# Verify installation
dorado --version
```

---

## Step 3: Download Basecalling Models

```bash
# Create models directory
mkdir -p ~/software/dorado_models

# Download R10.4.1 model (most common modern chemistry)
# Options: fast (fastest), hac (balanced), sup (most accurate)
dorado download --model dna_r10.4.1_e8.2_400bps_hac@v5.0.0 --models-directory ~/software/dorado_models

# For methylation calling, also download modification models:
dorado download --model dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v2 --models-directory ~/software/dorado_models
```

---

## Step 4: Install Python Dependencies

```bash
# Create conda environment
conda create -n nanopore python=3.10 -y
conda activate nanopore

# Install required packages
pip install uncalled4 pod5 pysam pandas numpy scipy

# Install samtools for BAM manipulation
conda install -c bioconda samtools -y
```

---

## Step 5: Basecall with Move Tables

The key flag is `--emit-moves` which outputs the signal-to-base mapping.

```bash
# Basic basecalling with move tables (GPU auto-detected)
dorado basecaller \
    ~/software/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
    /path/to/reads.pod5 \
    --emit-moves \
    --reference /path/to/reference.fasta \
    > output_with_moves.bam

# For CPU-only (much slower):
dorado basecaller \
    ~/software/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
    /path/to/reads.pod5 \
    --emit-moves \
    --reference /path/to/reference.fasta \
    -x cpu \
    > output_with_moves.bam

# With methylation calling:
dorado basecaller \
    ~/software/dorado_models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 \
    /path/to/reads.pod5 \
    --emit-moves \
    --reference /path/to/reference.fasta \
    --modified-bases 5mCG_5hmCG \
    > output_with_mods_and_moves.bam
```

---

## Step 6: Create Signal-Aligned BAM with uncalled4

uncalled4 uses DTW to refine signal-to-reference alignment.

```bash
# Run uncalled4 alignment
uncalled4 align \
    --ref /path/to/reference.fasta \
    --reads /path/to/reads.pod5 \
    --bam-in output_with_moves.bam \
    -o signal_aligned.bam \
    --basecaller-profile dna_r10.4.1_400bps \
    -p 8  # parallel processes
```

---

## Step 7: Verify Output

```bash
# Check for mv (move) and ts (stride) tags
samtools view signal_aligned.bam | head -1 | tr '\t' '\n' | grep -E "^(mv|ts):"

# Expected output:
# mv:B:c,1,0,0,1,0,0,0,1,...  (move table)
# ts:i:10                      (stride value)
```

---

## Quick Reference: File Formats

| File | Description |
|------|-------------|
| `.pod5` | Raw signal data (current measurements at 4-5 kHz) |
| `.bam` (standard) | Aligned reads, no signal info |
| `.bam` (with mv tag) | Aligned reads + move table mapping signal→bases |
| `.bam` (signal-aligned) | Full signal-to-reference alignment from uncalled4 |

---

## Troubleshooting

### "moves missing for read" error in uncalled4
Your BAM doesn't have `mv` tags. Re-basecall with `dorado --emit-moves`.

### "cannot find -lz" during pip install
Install zlib: `sudo apt-get install zlib1g-dev`

### Slow basecalling
- Use GPU: Remove `-x cpu` flag
- Use `fast` model instead of `hac` or `sup`
- Process files in parallel with `--batchsize`

### CUDA out of memory
Reduce batch size: `dorado basecaller ... --batchsize 64`

---

## Performance Expectations

| Mode | Speed (R10.4.1 hac) | Hardware |
|------|---------------------|----------|
| GPU (RTX 3090) | ~400 bases/sec/read | NVIDIA GPU |
| GPU (RTX 4090) | ~600 bases/sec/read | NVIDIA GPU |
| CPU (16 cores) | ~10 bases/sec/read | x86_64 |

For 5000 reads of 155bp each: GPU ~2 min, CPU ~2 hours.

---

## Completed Pipeline Run (2026-01-11)

The full signal alignment pipeline was executed successfully. Below are the results.

### Pipeline Execution Summary

| Step | Status | Details |
|------|--------|---------|
| Control basecalling | ✅ Complete | 5,295 reads basecalled with move tables |
| Control signal alignment | ✅ Complete | 2,617 reads aligned (2,885 too short) |
| 5mC basecalling | ✅ Complete | ~3,000 reads basecalled with move tables |
| 5mC signal alignment | ✅ Complete | 1,274 reads aligned (1,730 too short) |
| Signal extraction | ✅ Complete | 30,402 measurements at 8 cytosine positions |
| Statistical analysis | ✅ Complete | Significant C vs 5mC difference detected |
| HMM training data | ✅ Complete | Multiple output formats generated |

### Output Files (`output/`)

| File | Size | Description |
|------|------|-------------|
| `control_signal_aligned.bam` | 2.3 MB | Signal-aligned BAM - canonical C (2,617 reads) |
| `5mc_signal_aligned.bam` | 1.1 MB | Signal-aligned BAM - 5mC (1,274 reads) |
| `signal_at_cytosines.csv` | 3.4 MB | 30,402 current measurements at cytosine sites |
| `hmm_emission_params_circuit_board.json` | 2.9 KB | Gaussian params for HMM C/mC states |
| `hmm_emission_params_pomegranate.json` | 3.3 KB | Pomegranate-compatible emission params |
| `hmm_training_sequences.csv` | 701 KB | Per-read signal sequences for training |
| `classification_parameters.csv` | 1.1 KB | Threshold classifier parameters |
| `kmer_current_model.csv` | 45 KB | 5-mer context current distributions |

### Key Results: 5mC vs Canonical C Signal Differences

**Overall Statistics:**
```
Control (C):   mean = 812.61 pA, std = 99.73 pA, n = 20,546
Modified (5mC): mean = 849.28 pA, std = 101.24 pA, n = 9,856

Delta: +36.67 pA (5mC is 4.5% higher)
Cohen's d: 0.365 (small-medium effect)
Welch's t-test: p = 1.39e-189 (highly significant)
```

**Per-Position Differences:**

| Position | Control (pA) | 5mC (pA) | Delta (pA) | p-value |
|----------|--------------|----------|------------|---------|
| 38 | 890.56 | 938.40 | +47.84 | 2.76e-58 |
| 50 | 822.39 | 861.64 | +39.26 | 4.37e-26 |
| 62 | 816.52 | 853.11 | +36.59 | 2.42e-24 |
| 74 | 831.59 | 873.97 | +42.38 | 2.83e-50 |
| 86 | 792.68 | 828.21 | +35.53 | 6.63e-26 |
| 98 | 779.18 | 807.76 | +28.59 | 1.39e-19 |
| 110 | 772.96 | 800.81 | +27.85 | 2.53e-18 |
| 122 | 794.30 | 831.37 | +37.07 | 7.60e-32 |

**Conclusion:** 5mC cytosines produce consistently higher current levels than canonical C across all positions, with the strongest signal at position 38 (+48 pA). This difference is suitable for HMM-based modification detection.

### Scripts Created

| Script | Purpose |
|--------|---------|
| `run_signal_alignment.sh` | End-to-end pipeline: POD5 → Dorado → uncalled4 → BAM |
| `extract_signal_at_positions.py` | Extract current levels at BED-defined positions |
| `compare_signals.py` | Statistical comparison and HMM data export |
| `build_hmm_training_data.py` | Generate HMM training files in multiple formats |

### How to Reproduce

```bash
conda activate nanopore

# Step 1: Run signal alignment (already done)
./run_signal_alignment.sh \
    filtered_pod_files/control1_filtered_adapter_01.pod5 \
    nanopore_ref_data/all_5mers.fa \
    output/control_signal

./run_signal_alignment.sh \
    filtered_pod_files/5mc_filtered_adapter_01.pod5 \
    nanopore_ref_data/all_5mers.fa \
    output/5mc_signal

# Step 2: Extract signals at cytosine positions
python extract_signal_at_positions.py \
    --control-bam output/control_signal_aligned.bam \
    --control-pod5 filtered_pod_files/control1_filtered_adapter_01.pod5 \
    --modified-bam output/5mc_signal_aligned.bam \
    --modified-pod5 filtered_pod_files/5mc_filtered_adapter_01.pod5 \
    --bed nanopore_ref_data/all_5mers_C_sites.bed \
    --output output/signal_at_cytosines.csv

# Step 3: Compare and generate HMM training data
python compare_signals.py \
    --input output/signal_at_cytosines.csv \
    --bed nanopore_ref_data/all_5mers_C_sites.bed

python build_hmm_training_data.py \
    --input output/signal_at_cytosines.csv \
    --bed nanopore_ref_data/all_5mers_C_sites.bed \
    --reference nanopore_ref_data/all_5mers.fa \
    --output-dir output
```

---

## 3-Way Classification Pipeline (2026-01-12)

The full truth set from ONT (`nanopore_ref_data/`) was processed for 3-way classification: C vs 5mC vs 5hmC.

### Pipeline Execution Summary

| Step | Status | Details |
|------|--------|---------|
| Control basecalling | ✅ Complete | 159,994 reads basecalled (RTX 3080 GPU) |
| Control signal alignment | ✅ Complete | 65,506 reads aligned (89,457 too short) |
| 5mC basecalling | ✅ Complete | 159,989 reads basecalled |
| 5mC signal alignment | ✅ Complete | 33,782 reads aligned (66,824 too short) |
| 5hmC basecalling | ✅ Complete | 159,941 reads basecalled |
| 5hmC signal alignment | ✅ Complete | 37,754 reads aligned (45,574 too short) |
| Signal extraction | ✅ Complete | 1,047,178 measurements at 8 cytosine positions |
| HMM training data | ✅ Complete | 3-way emission parameters generated |

### Output Files (`output/rep1/`)

| File | Size | Description |
|------|------|-------------|
| `control_signal_aligned.bam` | 54 MB | Signal-aligned BAM - canonical C (65,506 reads) |
| `5mC_signal_aligned.bam` | 28 MB | Signal-aligned BAM - 5mC (33,782 reads) |
| `5hmC_signal_aligned.bam` | 36 MB | Signal-aligned BAM - 5hmC (37,754 reads) |
| `signal_at_cytosines_3way.csv` | 115 MB | 1,047,178 current measurements at cytosine sites |
| `hmm_3way_circuit_board.json` | 3.2 KB | Gaussian params for HMM C/mC/hmC states |
| `hmm_3way_pomegranate.json` | 5.2 KB | Pomegranate-compatible emission params |

### Key Results: 3-Way Signal Differences

**Overall Statistics:**
```
Control (C):   mean = 800.47 pA, std = 108.19 pA, n = 513,290
5mC:           mean = 830.72 pA, std = 108.84 pA, n = 264,300
5hmC:          mean = 809.41 pA, std = 113.49 pA, n = 269,588

Delta C→5mC:  +30.25 pA (3.8% increase)
Delta C→5hmC: +8.94 pA (1.1% increase)
Delta 5mC→5hmC: -21.31 pA (5mC higher than 5hmC)
```

**Per-Position Mean Currents (pA):**

| Position | Control (C) | 5mC | 5hmC | C→5mC | C→5hmC |
|----------|-------------|-----|------|-------|--------|
| 38 | 866.9 | 893.6 | 856.9 | +26.7 | -10.0 |
| 50 | 816.7 | 846.7 | 826.8 | +30.0 | +10.1 |
| 62 | 794.8 | 823.2 | 810.9 | +28.4 | +16.1 |
| 74 | 799.9 | 827.9 | 812.3 | +28.0 | +12.3 |
| 86 | 789.8 | 821.6 | 798.8 | +31.7 | +9.0 |
| 98 | 777.7 | 808.1 | 789.7 | +30.4 | +12.0 |
| 110 | 777.7 | 809.2 | 787.4 | +31.6 | +9.8 |
| 122 | 779.3 | 814.1 | 791.4 | +34.8 | +12.1 |

**Conclusion:**
- **5mC** produces consistently higher current (+27-35 pA) than canonical C
- **5hmC** produces intermediate current levels (+9-16 pA above C, ~20 pA below 5mC)
- Position 38 is anomalous (5hmC lower than C), may be context-dependent
- Clear separation enables 3-way HMM classification

### Scripts Created

| Script | Purpose |
|--------|---------|
| `extract_signal_3samples.py` | Extract signals from 3 BAM/POD5 pairs |
| `build_hmm_training_data_3way.py` | Generate 3-way HMM emission parameters |

### How to Reproduce

```bash
conda activate nanopore

# Step 1: Run signal alignment on all 3 samples
./run_signal_alignment.sh \
    nanopore_ref_data/control_rep1.pod5 \
    nanopore_ref_data/all_5mers.fa \
    output/rep1/control

./run_signal_alignment.sh \
    nanopore_ref_data/5mC_rep1.pod5 \
    nanopore_ref_data/all_5mers.fa \
    output/rep1/5mC

./run_signal_alignment.sh \
    nanopore_ref_data/5hmC_rep1.pod5 \
    nanopore_ref_data/all_5mers.fa \
    output/rep1/5hmC

# Step 2: Extract signals at cytosine positions
python extract_signal_3samples.py \
    --control-bam output/rep1/control_signal_aligned.bam \
    --control-pod5 nanopore_ref_data/control_rep1.pod5 \
    --5mC-bam output/rep1/5mC_signal_aligned.bam \
    --5mC-pod5 nanopore_ref_data/5mC_rep1.pod5 \
    --5hmC-bam output/rep1/5hmC_signal_aligned.bam \
    --5hmC-pod5 nanopore_ref_data/5hmC_rep1.pod5 \
    --bed nanopore_ref_data/all_5mers_C_sites.bed \
    --output output/rep1/signal_at_cytosines_3way.csv

# Step 3: Generate HMM training data
python build_hmm_training_data_3way.py \
    --input output/rep1/signal_at_cytosines_3way.csv \
    --output-dir output/rep1
```

---

## Next Steps

### Use the 3-Way HMM Training Data

```python
import json

# Load 3-way emission parameters
with open('output/rep1/hmm_3way_circuit_board.json') as f:
    params = json.load(f)

# Each position has C, mC, and hmC Gaussian distributions
for dist in params['distributions']:
    pos = dist['position']
    c_mean, c_std = dist['C']['mean'], dist['C']['std']
    mc_mean, mc_std = dist['mC']['mean'], dist['mC']['std']
    hmc_mean, hmc_std = dist['hmC']['mean'], dist['hmC']['std']
    print(f"Position {pos}:")
    print(f"  C={c_mean:.1f}±{c_std:.1f}")
    print(f"  mC={mc_mean:.1f}±{mc_std:.1f} (Δ={mc_mean-c_mean:+.1f})")
    print(f"  hmC={hmc_mean:.1f}±{hmc_std:.1f} (Δ={hmc_mean-c_mean:+.1f})")
```

### Train the Full Circuit Board HMM

See `UCSCN/epigenetics.py` for the full HMM implementation with artifact handling states.

---

## Related Documentation

- [`nanopore_ref_data/DATA_SUMMARY.md`](nanopore_ref_data/DATA_SUMMARY.md) - Full truth set documentation
- [`nanopore_ref_data/kmer_models/`](nanopore_ref_data/kmer_models/) - K-mer current tables and HMM building guide
- [`CLAUDE.md`](CLAUDE.md) - Project overview and HMM architecture
- [`refrences/hackathon_plans.md`](refrences/hackathon_plans.md) - Hackathon track plans
