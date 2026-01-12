# Building HMM Models for Nanopore Methylation Detection

## Table of Contents
1. [Overview](#overview)
2. [Original Schreiber-Karplus Model](#original-schreiber-karplus-model)
3. [Understanding build_profile()](#understanding-build_profile)
4. [R10.4.1 Chemistry Differences](#r104-chemistry-differences)
5. [K-mer Model Resources](#k-mer-model-resources)
6. [Modified Base Detection](#modified-base-detection)
7. [Building a Custom Model](#building-a-custom-model)
8. [Your Specific Sequence](#your-specific-sequence)

---

## Overview

This document summarizes findings for adapting the Schreiber-Karplus HMM-based methylation detection approach to modern Oxford Nanopore R10.4.1 chemistry.

**Key Challenge**: The original model was designed for:
- Older nanopore chemistry with ~5-nucleotide sensing window
- A specific hairpin-primer experimental design (reads template twice)
- Hand-curated training data from controlled experiments

Modern R10.4.1 requires significant adaptation.

---

## Original Schreiber-Karplus Model

### Experimental Design

The original experiment used a **bidirectional read** strategy:

```
Phase 1 (Unzipping): 5'→3' as blocking oligomer is removed
Phase 2 (Synthesis): 3'→5' as polymerase replicates

         ┌─────────── Template Strand ───────────┐
         5' CAT-CAT-CAT─CCGG─CAT-CAT...LABEL...CAT 3'
                         ↑
                    Modification site (CpG)

The same modification is read TWICE with opposite orientation.
```

### HMM Architecture

The model uses a modular "circuit board" design with **7 ports** per module:

| Component | Purpose |
|-----------|---------|
| **D-Module** | One per template position, handles match/artifact states |
| **U-Module** | Between D-modules, handles undersegmentation |
| **Fork Structure** | Parallel paths for C/mC/hmC at modification sites |

**State Types in Each D-Module**:

| State | Distribution | Purpose |
|-------|-------------|---------|
| M | Normal(μᵢ, σᵢ) | Primary match to position i |
| MO | Normal(μᵢ, σᵢ) + 80% self-loop | Oversegmentation handling |
| I | Uniform(0, 90) | Insert/noise handling |
| D | Silent | Deletion (position skipped) |
| MS/ME | Normal(μᵢ, σᵢ) | Flicker/stutter states |

**Fork Structure for Classification**:
```
         ┌─ C   → Normal(μ_C, σ_C)   ──┐
    ─────┼─ mC  → Normal(μ_mC, σ_mC) ──┼─────
         └─ hmC → Normal(μ_hmC, σ_hmC)─┘

Each path has DIFFERENT emission distribution.
Path with highest probability flow = classification.
```

---

## Understanding build_profile()

The `build_profile()` function in `epigenetics.py` constructs a list where:
- **Distribution object** → Linear position (single emission path)
- **Dictionary** → Fork position (multiple parallel paths)

### Position-by-Position Breakdown

| Code | Positions | Count | Description |
|------|-----------|-------|-------------|
| `dists['CAT'][::-1]` | 1-3 | 3 | Unzipping start, REVERSED |
| `{C, mC, hmC}` loop [8→0] | 4-12 | 9 | First cytosine fork, REVERSED indices |
| `dists['CAT'][::-1]` | 13-15 | 3 | Background (unzip phase) |
| `dists['CAT'][::-1]` | 16-18 | 3 | Background (unzip phase) |
| `dists['CAT']` | 19-21 | 3 | Background (synthesis begins, FORWARD) |
| `dists['CAT']` | 22-24 | 3 | Background |
| `{C, mC, hmC}` loop [0→8] | 25-33 | 9 | Second cytosine fork, FORWARD indices |
| `dists['CAT']` | 34-36 | 3 | Background |
| `{T, X, CAT}` loop | 37-42 | 6 | Label fork (validation marker) |
| `dists['CAT']` × 4 | 43-54 | 12 | Background to end |

**Total: 54 positions**

### Why 9 Positions Per Modification Site?

The nanopore has a ~5-nucleotide sensing zone. When reading a modification:

```
Template: ...CAT-C-C-G-G-CAT...
                ↑
           modification here

As this slides through the ~5nt pore window, the modification
influences ~9 consecutive segment readings (4 before + 1 at + 4 after)
```

### Why Reversed vs Forward?

```python
dists['CAT'][::-1]  # Unzip phase: reads 5'→3' (backwards through pore)
dists['CAT']        # Synthesis phase: reads 3'→5' (forwards through pore)

# Cytosine fork indices also flip:
dists['C'][8-i]     # First reading (unzip) - reversed
dists['C'][i]       # Second reading (synthesis) - forward
```

---

## R10.4.1 Chemistry Differences

### Key Specifications

| Parameter | Old Chemistry | R10.4.1 E8.2 |
|-----------|--------------|--------------|
| **K-mer size** | 5-6 mer | **9-mer** |
| **Sensing architecture** | Single reader | **Dual reader head** |
| **Sample rate** | Variable | 5 kHz |
| **Translocation speed** | Variable | 400 bps |
| **Total k-mers** | 4,096 (6-mer) | **262,144 (9-mer)** |

### Dual Reader Head Architecture

```
R10.4.1 pore cross-section:

    ════════════════════
         │      │
         │  R1  │  ← Reader 1 (positions 6-7) - PRIMARY
         │      │
    ─────┴──────┴─────
         │      │
         │  R2  │  ← Reader 2 (positions 1-5) - Secondary
         │      │
    ════════════════════

9-mer positions:  1  2  3  4  5  6  7  8  9
                  └────────┘  └────────┘
                  Secondary   PRIMARY
                  (weak)      (STRONG)
```

**Critical Finding**:
- Modifications at positions 6-7 → **Most divergent current** (detectable)
- Modifications at positions 1-5 → **Almost no signal** (not detectable)

---

## K-mer Model Resources

### Official ONT Canonical Model

**Repository**: https://github.com/nanoporetech/kmer_models

**Direct Download**:
```bash
wget https://raw.githubusercontent.com/nanoporetech/kmer_models/master/dna_r10.4.1_e8.2_400bps/9mer_levels_v1.txt
```

**Format**:
```
AAAAAAAAA    -1.8424    ← 9-mer + z-scored current level
AAAAAAAAC    -1.6519
...
262,144 total lines
```

**Limitation**: Contains only canonical bases (A, C, G, T). No 5mC or 5hmC.

### Resource Summary

| Resource | URL | K-mer Size | Modified Bases | Format |
|----------|-----|-----------|----------------|--------|
| ONT kmer_models | [GitHub](https://github.com/nanoporetech/kmer_models) | 9-mer | No | TXT |
| Uncalled4 | [GitHub](https://github.com/skovaka/uncalled4) | 9-mer | Trainable 5mC | NPZ/TSV |
| Dorado | [GitHub](https://github.com/nanoporetech/dorado) | Neural net | 5mC, 5hmC, 6mA | Model files |
| Poregen | [GitHub](https://github.com/hiruna72/poregen) | Configurable | Custom | TSV |
| f5c | [GitHub](https://github.com/hasindu2008/f5c) | 9-mer | 5mC | Inbuilt |

### Dorado Modified Base Models

Available for R10.4.1 E8.2 400bps:

| Model | Description |
|-------|-------------|
| `5mCG_5hmCG` | 5mC and 5hmC in CpG context |
| `5mC_5hmC` | 5mC and 5hmC in all contexts |
| `5mC` | 5mC in all contexts |
| `6mA` | N6-methyladenosine |
| `4mC_5mC` | 4mC and 5mC detection |

```bash
dorado download --model dna_r10.4.1_e8.2_400bps_sup@v5.2.0
```

---

## Modified Base Detection

### 5mC Current Shift Characteristics

From [Uncalled4 (Nature Methods 2025)](https://www.nature.com/articles/s41592-025-02631-4):

- Trained 9-mer models on M.SssI-treated D. melanogaster DNA (88% CpG methylation)
- Compared current levels between methylated and unmodified models

**Key Findings**:
1. K-mers with CpG at central position (6-7) show **most divergent** currents
2. CpGs at positions 1-5 provide **almost no modification information**
3. Typical current shift for 5mC: **0.5-1.5 z-score units** in CpG context

### Detection Methods

| Tool | Method | Strengths |
|------|--------|-----------|
| Uncalled4 | K-S test / z-scores on current levels | Higher sensitivity, trainable |
| f5c | Current level comparison | Faster, simpler |
| Dorado | Neural network | Integrated, production-ready |

---

## Building a Custom Model

### Conceptual Profile Builder

```python
def build_profile_custom(sequence, mod_positions, window=9):
    """
    Build HMM profile for arbitrary sequence with modifications.

    Args:
        sequence: DNA string
        mod_positions: dict like {74: ['C', 'mC', 'hmC'], 86: ['C', 'mC']}
        window: sensing window size (9 for R10.4.1)

    Returns:
        List of distributions/dicts for EpigeneticsModel()
    """
    profile = []
    half_window = window // 2

    for pos in range(len(sequence)):
        # Check if any modification affects this position
        affecting_mods = []
        for mod_pos, variants in mod_positions.items():
            if abs(pos - mod_pos) <= half_window:
                affecting_mods.append((mod_pos, variants))

        if affecting_mods:
            # This position needs a FORK
            profile.append({
                variant: get_distribution(sequence, pos, variant)
                for variant in ['C', 'mC', 'hmC']  # or subset
            })
        else:
            # Linear position - single distribution
            profile.append(get_distribution(sequence, pos, 'canonical'))

    return profile
```

### Getting Distributions

**Option 1**: Look up from 9mer_levels_v1.txt (canonical only)
```python
def get_canonical_level(kmer_table, sequence, pos):
    """Get expected current for 9-mer centered at position."""
    kmer = sequence[pos-4:pos+5]  # 9-mer centered at pos
    return kmer_table[kmer]
```

**Option 2**: Train custom model with Poregen
```bash
poregen collect \
    -i signal.blow5 \
    -s reads.fasta \
    -a alignment.paf \
    -k 9 \
    --sample_limit 5000 \
    -o custom_9mer_model.tsv
```

**Option 3**: Use Uncalled4 to train modification-aware model
```bash
uncalled4 train \
    --reads signal.pod5 \
    --reference sequence.fa \
    --model-preset dna_r10.4.1_400bps_9mer \
    --output custom_model.npz
```

---

## Your Specific Sequence

### Sequence Details

```
CCTGTACTTCGTTCAGTTACGTATTGCTGTGGAGTGTTCCAGCCACCTCACCGCTAAAGTCGCACAGCAA
TCTTCCCGTCATCACACGCGAGATCGCGCGCGTGAAACGTCAGAGCATCACTCTATAAGCAATACGTAAC
TGAACGAAGTACAGG
```

**Length**: 155 bp

### Modification Sites (0-indexed)

| Position | Base | Modification | Context |
|----------|------|--------------|---------|
| 74 | C | hmC | ...TCTT**C**CCGT... |
| 86 | C | mC | ...ATCA**C**ACGC... |
| 98 | C | mC | ...AGAT**C**GCGC... |

### K-mer Context Analysis

For each modification to be detectable, it should fall at positions 6-7 in the 9-mer:

```
Position 74 (hmC):
  When C is at k-mer position 7:
  9-mer = AATCTTCCC (positions 69-77)
  ✓ Modification at primary reader head

Position 86 (mC):
  When C is at k-mer position 7:
  9-mer = TCATCACAC (positions 80-88)
  ✓ Modification at primary reader head

Position 98 (mC):
  When C is at k-mer position 7:
  9-mer = CGAGATCGC (positions 92-100)
  ✓ Modification at primary reader head
```

**All three modifications are positioned for optimal detection.**

### Sequencing Platform

From metadata.txt:
- **Chemistry**: R10.4.1 E8.2
- **Kit**: SQK-LSK114
- **Flow cell**: FLO-PRO114M (PromethION)
- **Speed**: 400 bps @ 5kHz
- **Experiment**: ID_5mer_all_context_truth_sets
- **Sample**: control_rep2

---

## Next Steps

1. **Download canonical k-mer table** (done: `9mer_levels_v1.txt`)
2. **Obtain modified base currents** via:
   - Uncalled4 training on labeled data
   - Poregen de novo model generation
   - Published 5mC/hmC current deltas
3. **Adapt `EpigeneticsModel()`** for 9-mer context
4. **Build profile** for your specific sequence
5. **Train/validate** on your POD5 data

---

## References

### Papers
- [Uncalled4 - Nature Methods 2025](https://www.nature.com/articles/s41592-025-02631-4)
- [DeepPlant - Nature Communications 2025](https://www.nature.com/articles/s41467-025-58576-x)
- [Poregen - Bioinformatics 2025](https://academic.oup.com/bioinformatics/article/41/4/btaf111/8078598)

### Repositories
- [ONT kmer_models](https://github.com/nanoporetech/kmer_models)
- [Uncalled4](https://github.com/skovaka/uncalled4)
- [Dorado](https://github.com/nanoporetech/dorado)
- [Poregen](https://github.com/hiruna72/poregen)
- [f5c](https://github.com/hasindu2008/f5c)

### Original Schreiber Model
- [UCSC Nanopore Repository](https://github.com/UCSCNanopore/Data)
- Schreiber et al. methodology for cytosine modification detection
