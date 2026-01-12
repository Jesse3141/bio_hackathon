# K-mer Current Level Table

## File: `9mer_levels_v1.txt`

**Source**: [github.com/nanoporetech/kmer_models](https://github.com/nanoporetech/kmer_models)
**Chemistry**: R10.4.1 E8.2 400bps
**Total entries**: 262,144 (4^9 = all possible 9-mers)

---

## What This Table Represents

This is a **lookup table** mapping DNA sequence to expected ionic current:

```
Format:  KMER<tab>NORMALIZED_CURRENT_LEVEL

AAAAAAAAA    -1.8424    ← low current (more blockage)
AAAAAAAAC    -1.6519
...
TTTTTTTTT    +0.5237    ← higher current (less blockage)
```

Each row answers: **"When this 9-mer is in the pore, what current do we expect?"**

---

## Physical Meaning

```
                Nanopore Cross-Section

══════════════════════════════════════
│                                    │
│    DNA strand passing through      │
│         ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓         │
│        [A-A-A-A-A-A-A-A-A]         │  ← 9 nucleotides in sensing zone
│                                    │
══════════════════════════════════════
                  ↓
          Ionic current flows
          around the DNA
                  ↓
          Measured: ~X pA

Different k-mers block different amounts of current!
- Purines (A,G) = bulkier → more blockage → lower current
- Pyrimidines (C,T) = smaller → less blockage → higher current
```

The R10.4.1 pore has a **dual reader head** architecture, which is why it requires 9-mers (longer sensing region than older R9.4.1 which used 6-mers).

---

## The Numbers Are Z-Scores

Values are **normalized** (mean ≈ 0, std ≈ 1):

| Value | Meaning |
|-------|---------|
| **-1.84** | Very LOW current (1.84 std devs below mean) |
| **0.00** | Average current |
| **+1.18** | Higher current (1.18 std devs above mean) |

This normalization allows the table to work across different sequencing runs regardless of baseline current levels.

---

## Why 262,144 Rows?

```
4 bases × 9 positions = 4⁹ = 262,144 possible k-mers
```

Every possible 9-nucleotide combination of A, C, G, T has a pre-measured expected current level.

---

## How It's Used

When basecalling/analysis software observes a current measurement:

```python
import pandas as pd

# Load the table
kmer_table = {}
with open('9mer_levels_v1.txt', 'r') as f:
    for line in f:
        kmer, level = line.strip().split('\t')
        kmer_table[kmer] = float(level)

# Look up expected current for a sequence context
def get_expected_current(sequence, position):
    """Get expected current for 9-mer centered at position."""
    start = position - 4
    end = position + 5
    kmer = sequence[start:end]
    return kmer_table.get(kmer, None)

# Example: find which k-mer best explains an observed current
observed_current = -1.65

best_match = min(kmer_table.items(), key=lambda x: abs(x[1] - observed_current))
print(f"Best match: {best_match}")  # → ('AAAAAAAAC', -1.6519...)
```

---

## Limitation: Canonical Bases Only

**This table contains ONLY canonical bases (A, C, G, T).**

Modified bases produce different current levels:

| Base | Description | Current vs C |
|------|-------------|--------------|
| C | Canonical cytosine | baseline |
| 5mC | 5-methylcytosine | shifted ~0.5-1.5 z-score |
| 5hmC | 5-hydroxymethylcytosine | different shift pattern |

To detect modifications, you need:
1. A separate modified k-mer table, OR
2. The delta (shift) between canonical and modified currents
3. Training data from known modified sequences

---

## R10.4.1 Dual Reader Head

The R10.4.1 pore has two constriction points:

```
9-mer positions:  1  2  3  4  5  6  7  8  9
                  └────────┘  └────────┘
                  Secondary   PRIMARY
                  Reader      Reader
                  (weak)      (STRONG)
```

**For modification detection**:
- Modifications at positions **6-7** → Most divergent current (best detection)
- Modifications at positions **1-5** → Almost no signal (poor detection)

This means when designing experiments, ensure your modification falls at the PRIMARY reader head position in the k-mer context.

---

## Related Resources

| Resource | URL | Description |
|----------|-----|-------------|
| ONT kmer_models | [GitHub](https://github.com/nanoporetech/kmer_models) | Official source |
| Uncalled4 | [GitHub](https://github.com/skovaka/uncalled4) | Trainable models including 5mC |
| Dorado | [GitHub](https://github.com/nanoporetech/dorado) | Neural network basecaller with mod detection |
| Poregen | [GitHub](https://github.com/hiruna72/poregen) | Custom k-mer model generation |

---

## Download Command

```bash
wget https://raw.githubusercontent.com/nanoporetech/kmer_models/master/dna_r10.4.1_e8.2_400bps/9mer_levels_v1.txt
```
