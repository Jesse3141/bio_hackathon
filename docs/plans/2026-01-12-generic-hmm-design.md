# GenericHMM Design Document

**Date**: 2026-01-12
**Status**: Approved for implementation

---

## Overview

Create a `GenericHMM` class for 3-way cytosine modification classification (C / 5mC / 5hmC) using a single HMM with fork states at each cytosine position. The model uses empirical emission parameters pooled from all 32 adapter sequences.

---

## Architecture

### State Structure

- **8 cytosine positions**: 38, 50, 62, 74, 86, 98, 110, 122
- **3 modification states per position**: C, 5mC, 5hmC
- **Total**: 24 emitting states

```
State indexing:
  Position 0 (pos 38):  C=0,  5mC=1,  5hmC=2
  Position 1 (pos 50):  C=3,  5mC=4,  5hmC=5
  ...
  Position 7 (pos 122): C=21, 5mC=22, 5hmC=23
```

### Fork Structure at Each Position

```
                    ┌── M_C ────┐
From prev pos ──►───┤── M_5mC ──├───►── To next pos
                    └── M_5hmC ─┘
```

### Transition Probabilities

| Transition | Probability | Purpose |
|------------|-------------|---------|
| Self-loop | 0.15 | Handle nanopore dwelling/stuck |
| Forward (to i+1) | 0.80 | Normal progression |
| Skip (to i+2) | 0.05 | Handle skipped bases |

At each forward/skip transition, probability splits equally among 3 modification states (÷3).

### Emission Distributions

Gaussian distributions with parameters computed empirically from signal data:

| Class | Source | Expected Pattern |
|-------|--------|------------------|
| C | control_rep1.pod5 | Baseline (~812 pA mean) |
| 5mC | 5mC_rep1.pod5 | +36.7 pA shift |
| 5hmC | 5hmC_rep1.pod5 | TBD from data |

Parameters pooled across all 32 adapters to create robust priors.

---

## Data Pipeline

### Input Files

| File | Sample Type | Description |
|------|-------------|-------------|
| `nanopore_ref_data/control_rep1.pod5` | C | Canonical cytosine |
| `nanopore_ref_data/control_rep1.bam` | C | Aligned reads |
| `nanopore_ref_data/5mC_rep1.pod5` | 5mC | 5-methylcytosine |
| `nanopore_ref_data/5hmC_rep1.pod5` | 5hmC | 5-hydroxymethylcytosine |
| `nanopore_ref_data/5hmC_rep1.bam` | 5hmC | Aligned reads |

### BED Files (ground truth labels)

| File | Description |
|------|-------------|
| `all_5mers_C_sites.bed` | Positions with canonical C |
| `all_5mers_5mC_sites.bed` | Positions with 5mC |
| `all_5mers_5hmC_sites.bed` | Positions with 5hmC |

### Extraction Process

1. For each sample type (control, 5mC, 5hmC):
   - Run signal alignment if BAM doesn't exist
   - Extract current levels at 8 cytosine positions
   - Record: adapter_id, position, read_id, mean_current, std, dwell_time

2. Combine all measurements into single CSV
3. Compute per-class emission parameters (pooled across adapters)

---

## Class Interface

```python
class GenericHMM:
    """3-way fork HMM for C/5mC/5hmC classification."""

    POSITIONS = [38, 50, 62, 74, 86, 98, 110, 122]
    MODIFICATIONS = ['C', '5mC', '5hmC']

    def __init__(self, emission_params: Optional[Dict] = None):
        """Initialize with optional pre-computed emission parameters."""

    @classmethod
    def from_data(cls, signal_csv: str) -> 'GenericHMM':
        """Build model from extracted signal data CSV."""

    @classmethod
    def from_params_json(cls, params_path: str) -> 'GenericHMM':
        """Load from saved emission parameters."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GenericHMM':
        """Train on signal data X with modification labels y."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict modification class (0=C, 1=5mC, 2=5hmC) per read."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, 3) array of class probabilities."""

    def classify_read(self, signal: np.ndarray) -> Dict:
        """Get per-position posterior probabilities."""

    def save(self, path: str) -> None:
        """Save model to file."""

    @classmethod
    def load(cls, path: str) -> 'GenericHMM':
        """Load model from file."""
```

---

## Implementation Files

| File | Purpose |
|------|---------|
| `extract_all_signals.py` | Extract signals from all 32 adapters, all 3 sample types |
| `methylation_hmm/generic_hmm.py` | GenericHMM class implementation |

---

## Output Files

| File | Description |
|------|-------------|
| `output/all_signals.csv` | Combined signal data from all samples |
| `output/generic_hmm_params.json` | Computed emission parameters |
| `output/trained_generic_hmm.pt` | Trained model checkpoint |

---

## Success Criteria

1. Extract signals from all 32 adapters across C, 5mC, 5hmC samples
2. Compute distinct emission parameters for each modification class
3. GenericHMM achieves reasonable 3-class accuracy (target: >60%)
4. Model handles variable-length reads via self-loops and skips
