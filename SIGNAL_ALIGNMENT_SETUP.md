# Signal Alignment Pipeline Setup Guide

## Overview

This guide sets up the pipeline to create **signal-aligned BAM files** from nanopore POD5 data. These files contain move tables (`mv` tags) that map raw signal samples to basecalled nucleotides - essential for HMM-based modification detection.

```
POD5 (raw signal) → Dorado (basecall + emit-moves) → BAM with mv tags → uncalled4 (DTW alignment) → Signal-aligned BAM
```

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

## Next Steps

Once you have signal-aligned BAMs, you can:
1. Extract per-position current levels for HMM training
2. Compare signal characteristics between modified/unmodified samples
3. Build custom modification detection models
